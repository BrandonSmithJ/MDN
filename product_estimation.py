from pathlib import Path
from sklearn import preprocessing
from tqdm  import trange 
import numpy as np 

from .mdn   import MDN
from .meta  import get_sensor_bands, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, generate_config, using_feature, split_data
from .metrics import performance
from .plot_utils import plot_scatter
from .benchmarks import run_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, LogTransformer, RatioTransformer, BaggingColumnTransformer


def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None):
	''' 
	Estimate all target variables for the given x_test. If a model doesn't 
	already exist, creates a model with the given training data. 
	'''		
	wavelengths  = get_sensor_bands(args.sensor, args)
	store_scaler = lambda scaler, args=[], kwargs={}: (scaler, args, kwargs)

	# Note that additional x_scalers are added to the beginning of the pipeline (e.g. Robust( bagging( ratio(x) ) ))
	args.x_scalers = [
			store_scaler(preprocessing.RobustScaler),
	]
	args.y_scalers = [
		store_scaler(LogTransformer),
		store_scaler(preprocessing.MinMaxScaler, [(-1, 1)]),
	]

	# We only want bagging to be applied to the columns if there are a large number of feature (e.g. ancillary features included) 
	many_features = (x_train is not None and x_train.shape[1] > 20) or (x_test is not None and x_test.shape[1] > 20)

	# Add bagging to the columns (use a random subset of columns, excluding the first <n_wavelengths> columns from the process)
	if using_feature(args, 'bagging') and (using_feature(args, 'ratio') or many_features):
		n_extra = 0 if not using_feature(args, 'ratio') else RatioTransformer.n_features # Number of ratio features added
		args.x_scalers = [
			store_scaler(BaggingColumnTransformer, [len(wavelengths)], {'n_extra':n_extra}),
		] + args.x_scalers
	
	# Add additional features to the inputs
	if using_feature(args, 'ratio'):
		args.x_scalers = [
			store_scaler(RatioTransformer, [list(wavelengths)]),
		] + args.x_scalers

	model_path = generate_config(args, create=x_train is not None)	
	args.config_name = model_path.name

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
				est = model.predict(x_test[i:i+chunk_size], confidence_interval=None)
				partial_est.append( np.array(est, ndmin=3) )

			estimates.append( np.hstack(partial_est) )
			if hasattr(model, 'session'): model.session.close()

			if args.verbose and y_test is not None:
				median = np.median(np.stack(estimates, axis=1)[0], axis=0)
				labels = get_labels(wavelengths, output_slices, n_out=y_test.shape[1])
				for lbl, y1, y2 in zip(labels, y_test.T, median.T):
					print( performance(f'{lbl:>7s} Median', y1, y2) )
				print(f'--- Done round {round_num} ---\n')

	# Confidence bounds will contain [upper bounds, lower bounds] with the same shape as 
	# estimates) if a confidence_interval within (0,1) is passed into model.predict 
	if x_test is not None:
		estimates, *confidence_bounds = np.stack(estimates, axis=1)
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
	

def main():
	args = get_args()

	# If a file was given, estimate the product for the Rrs contained within
	if args.filename:
		filename = Path(args.filename)
		assert(filename.exists()), (
			f'Expecting path to in situ data as the passed argument, but "{filename}" does not exist.')

		x_test = np.loadtxt(args.filename, delimiter=',')
		print(f'Min Rrs: {x_test.min(0)}')
		print(f'Max Rrs: {x_test.max(0)}')
		print(f'Generating estimates for {len(x_test)} data points ({x_test.shape})')
		preds, idxs = get_estimates(args, x_test=x_test)
		print(f'Min: {np.median(preds, 0).min(0)}')
		print(f'Max: {np.median(preds, 0).max(0)}')

		labels = get_labels(get_sensor_bands(args.sensor, args), idxs, preds[0].shape[1])
		preds  = np.append(np.array(labels)[None,:], np.median(preds, 0), 0)

		filename = filename.parent.joinpath(f'MDN_{filename.stem}.csv').as_posix()
		print(f'Saving estimates at location "{filename}"')
		np.savetxt(filename, preds.astype(str), delimiter=',', fmt='%s')

	# Save data used with the given args
	elif args.save_data:
		x_data, y_data, slices, locs = get_data(args)

		valid  = np.any(np.isfinite(x_data), 1)
		x_data = x_data[valid].astype(str)
		y_data = y_data[valid].astype(str)
		locs   = np.array(locs).T[valid].astype(str)
		wvls   = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
		lbls   = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
		data   = np.append([wvls], x_data.astype(str), 0)
		data_full = np.append(np.append(locs, x_data, 1), y_data, 1)
		data_full = np.append([['index', 'dataset']+wvls+lbls], data_full, 0)
		np.savetxt(f'{args.sensor}_data_full.csv', data_full, delimiter=',', fmt='%s')

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

		estimates, est_slice = get_estimates(args, x_train, y_train, x_test, y_test, slices)
		estimates = np.median(estimates, 0)
		print('Shape Estimates:', estimates.shape)
		print('Min/Max Estimates:', get_minmax(estimates), '\n')

		labels     = get_labels(bands, slices, y_test.shape[1])
		products   = args.product.split(',') 
		benchmarks = run_benchmarks(args, args.sensor, x_test, y_test, {p:slices[p] for p in products}, silent=False, x_train=x_train, y_train=y_train, gridsearch=False, with_ml=False)
		benchmarks['MDN'] = estimates

		for p in products:
			for lbl, y1, y2 in zip(labels[slices[p]], y_test.T[slices[p]], estimates.T[slices[p]]):
				print( performance(f'MDN {lbl}', y1, y2) ) 
		print()
		plot_scatter(y_test, benchmarks, bands, labels, products, args.sensor)

	# Otherwise, train a model with all data (if not already existing)
	else:
		x_data, y_data, slices, locs = get_data(args)
		get_estimates(args, x_data, y_data, output_slices=slices)

