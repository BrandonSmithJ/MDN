from pathlib import Path
from sklearn import preprocessing
from tqdm  import trange 
import numpy as np 
import hashlib

from .mdn   import MDN
from .meta  import get_sensor_bands, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, generate_config, using_feature, split_data
from .metrics import performance
from .plot_utils import plot_scatter
from .benchmarks import run_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, generate_scalers


def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None, dataset_labels=None):
	''' 
	Estimate all target variables for the given x_test. If a model doesn't 
	already exist, creates a model with the given training data. 
	'''		
	# Add x/y scalers to the args object
	generate_scalers(args, x_train, x_test)

	# Add a few additional variables to be stored in the generated config file
	if x_train is not None: setattr(args, 'data_xtrain_shape', x_train.shape)
	if y_train is not None: setattr(args, 'data_ytrain_shape', y_train.shape)
	if x_test  is not None: setattr(args, 'data_xtest_shape',  x_test.shape)
	if y_test  is not None: setattr(args, 'data_ytest_shape',  y_test.shape)
	if dataset_labels is not None: 
		sets_str  = ','.join(sorted(map(str, np.unique(dataset_labels))))
		sets_hash = hashlib.sha256(sets_str.encode('utf-8')).hexdigest()
		setattr(args, 'datasets_hash', sets_hash)

	model_path = generate_config(args, create=x_train is not None)	
	args.config_name = model_path.name
	if args.verbose: print(model_path)

	uppers, lowers   = [], []
	x_full, y_full   = x_train, y_train
	x_valid, y_valid = None, None

	estimates = []
	for round_num in trange(args.n_rounds, disable=args.verbose or (args.n_rounds == 1) or args.silent):
		args.curr_round = round_num
		curr_round_seed = args.seed+round_num if args.seed is not None else None
		np.random.seed(curr_round_seed)

		# 75% of rows used in bagging
		if using_feature(args, 'bagging') and x_train is not None and args.n_rounds > 1:
			(x_train, y_train), (x_valid, y_valid) = split_data(x_full, y_full, n_train=0.75, seed=curr_round_seed) 

		datasets = {k: dict(zip(['x','y'], v)) for k,v in {
			'train' : [x_train, y_train],
			'valid' : [x_valid, y_valid],
			'test'  : [x_test, y_test],
			'full'  : [x_full, y_full],
		}.items() if v[0] is not None}

		model_kwargs = {
			'n_mix'      : args.n_mix, 
			'hidden'     : [args.n_hidden] * args.n_layers, 
			'lr'         : args.lr,
			'l2'         : args.l2,
			'n_iter'     : args.n_iter,
			'batch'      : args.batch,
			'avg_est'    : args.avg_est,
			'imputations': args.imputations,
			'epsilon'    : args.epsilon,
			'threshold'  : args.threshold,
			'scalerx'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.x_scalers]),
			'scalery'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.y_scalers]),
			'model_path' : model_path.joinpath(f'Round_{round_num}'),
			'no_load'    : args.no_load,
			'no_save'    : args.no_save,
			'seed'       : curr_round_seed,
			'verbose'    : args.verbose,
		}

		model = MDN(**model_kwargs)
		model.fit(x_train, y_train, output_slices, args=args, datasets=datasets)

		if x_test is not None:
			partial_est = []
			chunk_size  = args.batch * 100

			# To speed up the process and limit memory consumption, apply the trained model to the given test data in chunks
			for i in trange(0, len(x_test), chunk_size, disable=not args.verbose):
				est = model.predict(x_test[i:i+chunk_size], confidence_interval=getattr(args, 'CI', None))
				partial_est.append( np.array(est, ndmin=3) )

			estimates.append( np.hstack(partial_est) )
			if hasattr(model, 'session'): model.session.close()

			if args.verbose and y_test is not None:
				median = np.median(np.stack(estimates, axis=1)[0], axis=0)
				labels = get_labels(args.data_wavelengths, output_slices, n_out=y_test.shape[1])
				for lbl, y1, y2 in zip(labels, y_test.T, median.T):
					print( performance(f'{lbl:>7s} Median', y1, y2) )
				print(f'--- Done round {round_num} ---\n')

	# Confidence bounds will contain [upper bounds, lower bounds] with the same shape as 
	# estimates) if a confidence_interval within (0,1) is passed into model.predict 
	if x_test is not None:
		estimates, *confidence_bounds = np.stack(estimates, axis=1)
		if len(confidence_bounds):
			return estimates, model.output_slices, confidence_bounds
	return estimates, model.output_slices


def image_estimates(data, sensor='', product_name='chl', rhos=False, anc=False, **kwargs):
	''' 
	Takes any number of input bands (shaped [Height, Width]) and
	returns the products for that image, in the same shape. 
	Assumes the given bands are ordered by wavelength from least 
	to greatest, and are the same bands used to train the network.
	Supported products: {chl}

	rhos and anc models are not yet available.  
	'''
	valid_products = ['chl']

	if rhos:  sensor = sensor.replace('S2B','MSI') + '-rho'
	elif anc: sensor = sensor.replace('S2B','MSI')

	if isinstance(data, list):
		assert(all([data[0].shape == d.shape for d in data])), (
			f'Not all inputs have the same shape: {[d.shape for d in data]}')
		data = np.dstack(data)

	assert(sensor), (
		f'Must pass sensor name to image_estimates function')
	assert(sensor in SENSOR_LABEL), (
		f'Requested sensor {sensor} unknown. Must be one of: {list(SENSOR_LABEL.keys())}')
	assert(product_name in valid_products), (
		f'Requested product unknown. Must be one of {valid_products}')
	assert(len(data.shape) == 3), (
		f'Expected data to have 3 dimensions (height, width, feature). Found shape: {data.shape}')
	
	expected_features = len(get_sensor_bands(sensor)) + (len(ANCILLARY)+len(PERIODIC) if anc or rhos else 0)
	assert(data.shape[-1] == expected_features), (
		f'Got {data.shape[-1]} features; expected {expected_features} features for sensor {sensor}')
	
	args = get_args(kwargs, product=product_name, sensor=sensor)
	if rhos: 
		setattr(args, 'n_iter', 10000)
		setattr(args, 'model_lbl', 'l2gen_rhos-anc')
	elif anc:
		setattr(args, 'n_iter', 10000)
		setattr(args, 'model_lbl', 'l2gen_Rrs-anc')
		
	im_shape = data.shape[:-1] 
	im_data  = np.ma.masked_invalid(data.reshape((-1, expected_features)))
	im_mask  = np.any(im_data.mask, axis=1)
	im_data  = im_data[~im_mask]
	pred,idx = get_estimates(args, x_test=im_data)
	products = np.median(pred, 0) 
	product  = np.atleast_2d( products[:, idx[product_name]] )
	est_mask = np.tile(im_mask[:,None], (1, product.shape[1]))
	est_data = np.ma.array(np.zeros(est_mask.shape)*np.nan, mask=est_mask, hard_mask=True)
	est_data.data[~im_mask] = product
	return [p.reshape(im_shape) for p in est_data.T]


def apply_model(x_test, use_cmdline=True, **kwargs):
	''' Apply a model (defined by kwargs and default parameters) to x_test '''
	args = get_args(kwargs, use_cmdline=use_cmdline)
	preds, idxs = get_estimates(args, x_test=x_test)
	return np.median(preds, 0), idxs
	

def print_dataset_stats(**kwargs):
	''' Print datasets shape & min / max stats per feature '''
	label = kwargs.pop('label', '')
	for k, arr in kwargs.items():
		print(f'\n{label} {k.title()}'.strip()+'\n\t'.join(['']+[f'{k}: {v}' for k, v in {
			'Shape'   : arr.shape,
			'N Valid' : getattr(np.isfinite(arr).sum(0), 'min' if arr.shape[1] > 10 else 'tolist')(),
			'Min,Max' : list(zip(np.nanmin(arr, 0).round(2), np.nanmax(arr, 0).round(2))),
		}.items()]))


def main():
	args = get_args()

	# If a file was given, estimate the product for the Rrs contained within
	if args.filename:
		filename = Path(args.filename)
		assert(filename.exists()), f'Expecting "{filename}" to be path to Rrs data, but it does not exist.'

		bands = get_sensor_bands(args.sensor, args)
		if filename.is_file(): x_test = np.loadtxt(args.filename, delimiter=',')
		else:                  x_test, *_ = _load_datasets(['Rrs'], [filename], bands)

		print(f'Generating estimates for {len(x_test)} data points ({x_test.shape})')
		print_dataset_stats(rrs=x_test, label='Input')

		estimates, slices = get_estimates(args, x_test=x_test)
		estimates = np.median(estimates, 0)
		print_dataset_stats(estimates=estimates, label='MDN')

		labels    = get_labels(bands, slices, estimates.shape[1])
		estimates = np.append([labels], estimates, 0).astype(str)
		filename  = filename.parent.joinpath(f'MDN_{filename.stem}.csv').as_posix()
		
		print(f'Saving estimates at location "{filename}"')
		np.savetxt(filename, estimates, delimiter=',', fmt='%s')

	# Save data used with the given args
	elif args.save_data:
		x_data, y_data, slices, locs = get_data(args)

		valid  = np.any(np.isfinite(x_data), 1)
		x_data = x_data[valid].astype(str)
		y_data = y_data[valid].astype(str)
		locs   = np.array(locs)[valid].astype(str)
		wvls   = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
		lbls   = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
		data   = np.append([wvls], x_data.astype(str), 0)
		data_full = np.append(np.append(locs, x_data, 1), y_data, 1)
		data_full = np.append([['dataset', 'index']+wvls+lbls], data_full, 0)
		np.savetxt(f'{args.sensor}_data_full.csv', data_full, delimiter=',', fmt='%s')
		print(f'Saved data with shape {data_full.shape}')

	# Train a model with partial data, and benchmark on remaining
	elif args.benchmark:
		import matplotlib.pyplot as plt
		import matplotlib.gridspec as gridspec
		import matplotlib.ticker as ticker
		import matplotlib.patheffects as pe 
		import seaborn as sns 

		if args.dataset == 'sentinel_paper':
			setattr(args, 'fix_tchl', True)
			setattr(args, 'seed', 1234)

		np.random.seed(args.seed)
		
		bands   = get_sensor_bands(args.sensor, args)
		n_train = 0.5 if args.dataset != 'sentinel_paper' else 1000

		x_data, y_data, slices, locs         = get_data(args)
		(x_train, y_train), (x_test, y_test) = split_data(x_data, y_data, n_train=n_train, seed=args.seed)

		get_minmax = lambda d: list(zip(np.nanmin(d, 0).round(2), np.nanmax(d, 0).round(2)))
		print(f'\nShapes: x_train={x_train.shape}  x_test={x_test.shape}  y_train={y_train.shape}  y_test={y_test.shape}')
		print('Min/Max Train X:', get_minmax(x_train))
		print('Min/Max Train Y:', get_minmax(y_train))
		print('Min/Max Test X:', get_minmax(x_test))
		print('Min/Max Test Y:', get_minmax(y_test))
		print(f'Train valid: {np.isfinite(y_train).sum(0)}')
		print(f'Test valid: {np.isfinite(y_test).sum(0)}')
		print(f'Min/Max wavelength: {bands[0]}, {bands[-1]}\n')

		estimates, est_slice = get_estimates(args, x_train, y_train, x_test, y_test, slices, dataset_labels=locs[:,0])
		estimates = np.median(estimates, 0)
		print('Shape Estimates:', estimates.shape)
		print('Min/Max Estimates:', get_minmax(estimates), '\n')

		labels     = get_labels(bands, slices, y_test.shape[1])
		products   = args.product.split(',') 
		benchmarks = run_benchmarks(args.sensor, x_test, y_test, x_train, y_train, {p:slices[p] for p in products}, verbose=True, return_ml=False)
		benchmarks['MDN'] = estimates

		for p in products:
			for lbl, y1, y2 in zip(labels[slices[p]], y_test.T[slices[p]], estimates.T[slices[p]]):
				print( performance(f'MDN {lbl}', y1, y2) ) 
		print()
		plot_scatter(y_test, benchmarks, bands, labels, products, args.sensor)

	# Otherwise, train a model with all data (if not already existing)
	else:
		x_data, y_data, slices, locs = get_data(args)
		get_estimates(args, x_data, y_data, output_slices=slices, dataset_labels=locs[:,0])

