from pathlib import Path
from sklearn import preprocessing
from tqdm  import trange 
from collections import defaultdict as dd

import numpy as np 
import pickle as pkl 
import hashlib 

# try: from .Development.Removed.mdn_cpu import MDN
# except: 
from .model import MDN

# try:    from .mdn  import MDN
# except: from .mdn2 import MDN 
from .meta  import get_sensor_bands, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, generate_config, using_feature, split_data, _load_datasets
from .metrics import performance, mdsa, sspb, msa
from .plot_utils import plot_scatter
from .benchmarks import run_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, generate_scalers


def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None, dataset_labels=None, x_sim=None, y_sim=None, return_model=False, return_coefs=False):
	''' 
	Estimate all target variables for the given x_test. If a model doesn't 
	already exist, creates a model with the given training data. 
	'''		
	# Add x/y scalers to the args object
	generate_scalers(args, x_train, x_test)

	if args.verbose: 
		print(f'\nUsing {len(args.wavelengths)} wavelength(s) in the range [{args.wavelengths[0]}, {args.wavelengths[-1]}]')
		if x_train is not None: print_dataset_stats(x=x_train, label='Train')
		if y_train is not None: print_dataset_stats(y=y_train, label='Train')
		if x_test  is not None: print_dataset_stats(x=x_test,  label='Test')
		if y_test  is not None: print_dataset_stats(y=y_test,  label='Test')
	
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
	
	predict_kwargs = {
		'avg_est'             : getattr(args, 'avg_est', False),
		'threshold'           : getattr(args, 'threshold', None),
		'confidence_interval' : getattr(args, 'CI', None),
		'use_gpu'             : getattr(args, 'use_gpu', False),
		'return_coefs'        : True,
	}

	x_full, y_full   = x_train, y_train
	x_valid, y_valid = None, None

	outputs = dd(list)
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
			'sim'   : [x_sim, y_sim],
		}.items() if v[0] is not None}

		model_kwargs = {
			'n_mix'      : args.n_mix, 
			'hidden'     : [args.n_hidden] * args.n_layers, 
			'lr'         : args.lr,
			'l2'         : args.l2,
			'n_iter'     : args.n_iter,
			'batch'      : args.batch,
			'imputations': args.imputations,
			'epsilon'    : args.epsilon,
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

		if return_model:
			outputs['model'].append(model)

		if return_coefs:
			outputs['scalerx'].append(model.scalerx)
			outputs['scalery'].append(model.scalery)

		if x_test is not None:
			(estimates, *confidence), coefs = model.predict(x_test, **predict_kwargs)
			outputs['estimates'].append(estimates)

			if return_coefs:
				outputs['coefs'].append(coefs)

			if len(confidence):
				upper, lower = confidence
				outputs['upper_bound'].append(upper) 
				outputs['lower_bound'].append(lower)

			if args.verbose and y_test is not None:
				median = np.median(outputs['estimates'], axis=0)
				labels = get_labels(args.wavelengths, output_slices, n_out=y_test.shape[1])
				for lbl, y1, y2 in zip(labels, y_test.T, median.T):
					print( performance(f'{lbl:>7s} Median', y1, y2) )
				print(f'--- Done round {round_num} ---\n')

		if hasattr(model, 'session'): model.session.close()

	if len(outputs) == 1:
		outputs = list(outputs.values())[0]
	return outputs, model.output_slices


def apply_model(x_test, use_cmdline=True, **kwargs):
	''' Apply a model (defined by kwargs and default parameters) to x_test '''
	args = get_args(kwargs, use_cmdline=use_cmdline)
	preds, idxs = get_estimates(args, x_test=x_test)
	return np.median(preds, 0), idxs


def image_estimates(data, sensor=None, function=apply_model, rhos=False, anc=False, **kwargs):
	''' 
	Takes data of shape [Height, Width, Wavelengths] and returns the outputs of the 
	given function for that image, in the same [H, W] shape. 
	rhos and anc models are not yet available.  
	'''
	def ensure_feature_dim(v):
		if len(v.shape) == 1:
			v = v[:, None]
		return v 

	if isinstance(data, list):
		assert(all([data[0].shape == d.shape for d in data])), (
			f'Not all inputs have the same shape: {[d.shape for d in data]}')
		data = np.dstack(data)

	assert(sensor is not None), (
		f'Must pass sensor name to image_estimates function. Options are: {list(SENSOR_LABEL.keys())}')
	assert(sensor in SENSOR_LABEL), (
		f'Requested sensor {sensor} unknown. Must be one of: {list(SENSOR_LABEL.keys())}')
	assert(len(data.shape) == 3), (
		f'Expected data to have 3 dimensions (height, width, feature). Found shape: {data.shape}')
	
	expected_features = len(get_sensor_bands(sensor)) + (len(ANCILLARY)+len(PERIODIC) if anc or rhos else 0)
	assert(data.shape[-1] == expected_features), (
		f'Got {data.shape[-1]} features; expected {expected_features} features for sensor {sensor}')
	
	im_shape = data.shape[:-1] 
	im_data  = np.ma.masked_invalid(data.reshape((-1, data.shape[-1])))
	im_mask  = np.any(im_data.mask, axis=1)
	im_data  = im_data[~im_mask]
	estimate = function(im_data, sensor=sensor, **kwargs) if im_data.size else np.zeros((0, 1))

	# Need to handle function which return extra information (e.g. a dictionary mapping output feature slices)
	if isinstance(estimate, tuple):
		estimate, *remaining = estimate 

	estimate = ensure_feature_dim(estimate)
	est_mask = np.tile(im_mask[:,None], (1, estimate.shape[-1]))
	est_data = np.ma.array(np.zeros(est_mask.shape)*np.nan, mask=est_mask, hard_mask=True)
	est_data.data[~im_mask] = estimate
	est_data = est_data.reshape(im_shape + est_data.shape[-1:])

	# Let the user handle the extra information of the function they passed, if there was any
	if len(remaining):
		if len(remaining) == 1: 
			remaining = remaining[0]
		return est_data, remaining
	return est_data



def print_dataset_stats(**kwargs):
	''' Print datasets shape & min / max stats per feature '''
	label = kwargs.pop('label', '')
	for k, arr in kwargs.items():
		if arr is not None:
			arr = np.array(arr)
			print(f'\n{label} {k.title()}'.strip()+'\n\t'.join(['']+[f'{k}: {v}' for k, v in {
				'Shape'   : arr.shape,
				'N Valid' : getattr(np.isfinite(arr).sum(0), 'min' if arr.shape[1] > 10 else 'tolist')(),
				'Minimum' : list(np.nanmin(arr, 0).round(2)),
				'Maximum' : list(np.nanmax(arr, 0).round(2))
				# 'Min,Max' : list(zip(np.nanmin(arr, 0).round(2), np.nanmax(arr, 0).round(2))),
			}.items()]))


def generate_estimates(args, bands, x_train, y_train, x_test, y_test, slices, locs, verbose=True, cache=None, **kwargs):
	''' Helper function for generating MDN / benchmark estimates in a single
		dictionary, with the option to cache the results. '''
	if cache is not None:
		cache = Path(cache)
		assert(cache.suffix == '.pkl'), f'Must provide cache as path to destination .pkl file: {cache}'
		cache.parent.mkdir(parents=True, exist_ok=True)

		if cache.exists():
			with cache.open('rb') as f:
				return pkl.load(f)

	# estimates, slices = get_estimates(args, x_train, y_train, x_test, y_test, slices, dataset_labels=locs[:,0], **kwargs)
	# estimates = np.median(estimates, 0)
	products  = args.product.split(',') 

	# if verbose: 
	# 	print_dataset_stats(estimates=estimates, label='MDN')
	# 	print()

	# 	labels = get_labels(bands, slices, y_test.shape[1])
		
	# 	if 0: 
	# 		products += ['OHE']
	# 		labels += ['OHE']
	# 		slices['OHE'] = slice(1, None)
	# 		from sklearn.metrics import f1_score,roc_auc_score
	# 		print(f'F1: {f1_score(y_test[:, 1:].argmax(1), estimates[:, 1:].argmax(1), average="weighted"):.2f}')
	# 		e = estimates[:,1:]
	# 		e/= e.sum(1, keepdims=True)
	# 		try: print(f'ROCAUC: {roc_auc_score(y_test[:,1:], e):.2f}')
	# 		except: pass

	benchmarks = run_benchmarks(args.sensor, x_test, y_test, x_train=x_train, y_train=y_train, 
								slices={p:slices[p] for p in products}, verbose=False, 
								# return_ml=len(products) == 1, kwargs_ml={'gridsearch': False})
								return_ml=False and x_train is not None, kwargs_ml={'gridsearch': False})
	# for p in products: benchmarks[p].update({'MDN' : estimates[..., slices[p]]})

	if verbose: 
		for p in benchmarks:
			print(f'\n---------------------- {p} ----------------------')
			errs = {method: mdsa(y_test[:, slices[p]], est) for method, est in benchmarks[p].items()}
			keys = sorted(benchmarks[p].keys(), key=lambda k: errs[k], reverse=True)
			for method in keys:
				print( performance(method, y_test[:, slices[p]], benchmarks[p][method]) )
	assert(0)
	if cache is not None:
		with cache.open('wb') as f:
			pkl.dump(benchmarks, f)
	return benchmarks


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

		valid  = np.any(np.isfinite(x_data), 1) # Remove samples which are completely nan
		x_data = x_data[valid].astype(str)
		y_data = y_data[valid].astype(str)
		locs   = np.array(locs)[valid].astype(str)
		wvls   = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
		lbls   = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
		data_full = np.append(np.append(locs, y_data, 1), x_data, 1)
		data_full = np.append([['dataset', 'index']+lbls+wvls], data_full, 0)
		filename  = f'{args.sensor}_data_full.csv'
		np.savetxt(filename, data_full, delimiter=',', fmt='%s')
		print(f'Saved data with shape {data_full.shape} to {filename}')

	# Train a model with partial data, and benchmark on remaining
	elif args.benchmark:

		if args.dataset == 'sentinel_paper':
			setattr(args, 'fix_tchl', True)
			setattr(args, 'seed', 1234)

		np.random.seed(args.seed)
		
		bands   = get_sensor_bands(args.sensor, args)
		n_train = 0.5 if args.dataset != 'sentinel_paper' else 1000
		x_data, y_data, slices, locs = get_data(args)

		use_ohe = 0
		if use_ohe:
			from .transformers import DatasetFeatureTransformer 
			dft = DatasetFeatureTransformer(locs[:,0])
			x_data = dft.fit_transform(x_data)
			# y_data = dft.fit_transform(y_data)
		else: 
			from .transformers import IdentityTransformer
			dft = IdentityTransformer()
		
		# Leave-one-out cross-validation
		if args.LOO_CV:
			counts   = []
			results  = dd(lambda: dd(lambda: dd(list)))
			metrics  = [msa, mdsa, sspb]
			datasets = [name for name, count in zip(*np.unique(locs[:,0], return_counts=True)) if count > 0]
			print(f'Performing LOO validation with {len(datasets)} datasets')
			counts2 = {}
			frame = [['Dataset', 'Method', 'Product', 'Metric', 'Value']]
			for name in datasets:
				print(f'\n\n--------------------------------------------------\n{name}:')
				setattr(args, 'model_lbl', f'LOO/{name}')

				curr_set = locs[:,0] == name
				kwargs   = {
					'x_train' : x_data[~curr_set],
					'y_train' : y_data[~curr_set],
					'x_test'  : x_data[curr_set],
					'y_test'  : y_data[curr_set],
					'slices'  : slices,
					'locs'    : locs,
					'cache'   : None,# f'LOO_cache/{args.sensor}/{name}_newdata2.pkl',
				}

				# setattr(args, 'use_sim', True)
				# x, y, s, l = get_data(args)
				# setattr(args, 'use_sim', False)
				# print(x.shape, y.shape)
				# kwargs.update({
				# 	# 'x_train' : np.append(kwargs['x_train'], x, 0), 
				# 	# 'y_train' : np.append(kwargs['y_train'], y[:, s[args.product]], 0),
				# 	'x_sim': x, 
				# 	'y_sim': y[:, s[args.product]],
				# 	# 'x_test'  : x_data[y_data.flatten() <= 100],
				# 	# 'y_test'  : y_data[y_data.flatten() <= 100],
				# 	# 'x_sim' : dft.transform(dft.inverse_transform(kwargs['x_train']), zeros=True),
				# 	# 'y_sim' : kwargs['y_train'],
				# })

				counts.append(curr_set.sum())
				counts2[name] = curr_set.sum()
				benchmarks = generate_estimates(args, bands, **kwargs)

				# assert(0)
				for product in benchmarks:
					for label, est in benchmarks[product].items():
						for metric in metrics:
							if label == 'OC3':
								print(name, product, label)
								print(kwargs['y_test'][:,slices[product]].shape, est.shape)
								print(slices)
								print(kwargs['y_test'][:,slices[product]][0], est[0])
								print(kwargs['x_test'][0])
								print(mdsa(kwargs['y_test'][:,slices[product]].flatten(), est.flatten()))
								assert(0)
							try: val = metric(kwargs['y_test'][:,slices[product]].flatten(), est.flatten())
							except Exception as e: 
								val = np.nan
							results[label][product][metric.__name__].append(val)
							frame.append([name, label, product, metric.__name__, val])

			totals = {}
			for label in sorted(results.keys(), key=lambda k: np.nanmean(results[k]['chl']['MdSA']), reverse=True):
				a = max(map(len, results[label]['chl'].values()))
				print(f'\n{label} ({a} valid datasets)')
				for product in sorted(results[label]):
					if max(map(len, results[label][product].values())):
						print(f'Product: {product}')
						print(f'Mean     | ' + '  '.join([f'{k}: {np.nanmean(v):.1f}' for k,v in results[label][product].items()]))
						print(f'Median   | ' + '  '.join([f'{k}: {np.nanmedian(v):.1f}' for k,v in results[label][product].items()]))
						print(len(np.all(np.isfinite(np.array(list(results[label][product].values()))), 0)), len(counts))
						print(f'Weighted | ' + '  '.join([f'{k}: {np.nansum(np.array(v) * np.array(counts)/sum(np.all(np.isfinite(np.array(list(results[label][product].values()))), 0) * np.array(counts))):.1f}' for k,v in results[label][product].items()]))
						totals[(label, product)] = sum(np.all(np.isfinite(np.array(list(results[label][product].values()))), 0) * np.array(counts))
			
			import pandas as pd 
			import seaborn as sns
			import matplotlib.pyplot as plt 
			frame = pd.DataFrame(frame[1:], columns=frame[0])
			frame['count2'] = np.array([counts2[row['Dataset']] if np.isfinite(row['Value']) else 0 for _, row in frame.iterrows()])
			frame['count'] = np.array([counts2[row['Dataset']] for _, row in frame.iterrows()])
			frame['total'] = np.array([totals[(row['Method'], row['Product'])] for _, row in frame.iterrows()])
			frame['weight'] = np.array([counts2[row['Dataset']] / totals[(row['Method'], row['Product'])] for _, row in frame.iterrows()])
			print(frame[(frame['Method'] == 'MDN') & (frame['Product'] == 'chl') & (frame['Metric'] == 'MSA')])
			print(frame[(frame['Method'] == 'MDN') & (frame['Product'] == 'chl') & (frame['Metric'] == 'MSA')].sum(0))

			assert(0)
			# frame.to_csv('LOO_results.csv', index=False)
			v = frame.copy()
			v['Value'] *= np.array([counts2[row['Dataset']] / totals[(row['Method'], row['Product'])] for _, row in v.iterrows()])
			v = v.groupby(['Method', 'Product', 'Metric']).sum()#.sort_values(['Product', 'Metric', 'Value'])#.to_csv('weighted_results.csv')
			v['Median'] = frame.groupby(['Method', 'Product', 'Metric']).median()
			v['Mean'] = frame.groupby(['Method', 'Product', 'Metric']).mean()
			v['Weighted'] = v['Value']
			v = v.drop(columns=['Value'])
			v.sort_values(['Product', 'Metric', 'Median']).to_csv('averaged_results.csv')
			# frame.groupby(['Method', 'Product', 'Metric']).mean().sort_values(['Product', 'Metric', 'Value']).to_csv('mean_results.csv')
			# frame.groupby(['Method', 'Product', 'Metric']).median().sort_values(['Product', 'Metric', 'Value']).to_csv('median_results.csv')
			assert(0)
			frame = frame[~frame['Method'].isin(['Gurlin_3band', 'OCx', 'GIOP', 'Mishra_NDCI', 'FLH', 'OC2', 'Novoa_old', 'SOLID_old', 'Mishra_modelled_NDCI', 'Gurlin_2band', 'Gilerson_2band', 'Moses_2band', 'OC3', 'Gons_2band'])]
			frame = frame[~frame['Dataset'].isin(['GreenBay2013'])]

			print(frame)
			# f, ax = plt.subplots(3,1, figsize=(15, 15))
			# for i, product in enumerate(['chl', 'tss', 'cdom']):
			# 	sns.barplot(data=frame[(frame['Product'] == product) & (frame['Metric'] == 'MdSA')], y='Dataset', x='Value', hue='Method', ax=ax[i])
			# 	ax[i].set_title(product)
			# 	ax[i].set_xlabel('')
			# 	ax[i].set_xscale('log')
			# ax[-1].set_xlabel('MdSA')
			# plt.show()

			for product in ['chl', 'tss', 'cdom']:
				f, ax = plt.subplots(1,1, figsize=(15, 10))
				f = frame[(frame['Product'] == product) & (frame['Metric'] == 'MdSA')]
				# if product == 'chl':
				# 	f = f[~f['Dataset'].isin(['CedricCABay', 'CedricGoM', 'CedricPlumIsland', 'GreenBay2013', 'Hubert', 'SeaSWIR'])]
				# else:
				all_nan = f.groupby('Dataset').sum() == 0
				f = f[~f['Dataset'].isin(all_nan.index.to_numpy()[all_nan.values.flatten()])]
				sns.barplot(data=f, y='Dataset', x='Value', hue='Method', ax=ax)
				ax.set_title(product)
				# ax.set_xscale('log')
				ax.set_xlabel('MdSA')
				plt.show()
		else:
			(x_train, y_train), (x_test, y_test) = split_data(x_data, y_data, n_train=n_train, seed=args.seed)

			# if use_ohe:
			# 	y_test = dft.transform(dft.inverse_transform(y_test)) # Replace set labels with zero vectors
			benchmarks = generate_estimates(args, bands, x_train, y_train, x_test, y_test, slices, locs)
			labels     = get_labels(bands, slices, y_test.shape[1])
			products   = args.product.split(',')
			plot_scatter(y_test, benchmarks, bands, labels, products, args.sensor)

	# Otherwise, train a model with all data (if not already existing)
	else:
		x_data, y_data, slices, locs = get_data(args)
		get_estimates(args, x_data, y_data, output_slices=slices, dataset_labels=locs[:,0])
