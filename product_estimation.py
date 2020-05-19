import numpy as np 
import pickle as pkl 
import argparse
import sys
import warnings
import hashlib

from collections import defaultdict as dd
from pathlib import Path
from sklearn import preprocessing
from glob  import glob 
from tqdm  import trange, tqdm 

from .mdn   import MDN
from .meta  import get_sensor_bands, get_sensor_label, SENSOR_LABEL
from .utils import add_identity, get_labels, get_data, find_wavelength, generate_config, bagging_subset, store_scaler, add_stats_box
from .metrics import performance, rmse, rmsle, leqznan, mape, r_squared, slope, nrmse, mwr, bias, mae
from .benchmarks import print_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, LogTransformer, NegLogTransformer, RatioTransformer, BaggingColumnTransformer


def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None):
	''' 
	Estimate all target variables for the given x_test. If a model doesn't 
	already exist, creates a model with the given training data. 
	'''

	if x_train is not None:
		x_train_orig = x_train.copy()
		y_train_orig = y_train.copy()

	wavelengths   = get_sensor_bands(args.sensor, args)
	using_bagging = (hasattr(args, 'no_bagging') and not args.no_bagging) or (hasattr(args, 'bagging') and args.bagging) 
	using_ratio   = (hasattr(args, 'no_ratio') and not args.no_ratio) or (hasattr(args, 'add_ratio') and args.add_ratio)

	args.x_scalers = [
			store_scaler(preprocessing.RobustScaler),
			
			# store_scaler(LogTransformer),
			# store_scaler(preprocessing.QuantileTransformer, [], {'output_distribution':'normal'}),
	]
	args.y_scalers = [
		store_scaler(LogTransformer),
		store_scaler(preprocessing.MinMaxScaler, [(-1, 1)]),
	]


	if using_bagging and (using_ratio or (x_train is not None and x_train.shape[1] > 20) or (x_test is not None and x_test.shape[1] > 20)):
		args.x_scalers = [
			store_scaler(BaggingColumnTransformer, [len(wavelengths) + len(get_sensor_bands(args.sensor.replace('-rho',''), args))]),
		] + args.x_scalers
	
	if using_ratio:
		args.x_scalers = [
			store_scaler(RatioTransformer, [list(wavelengths)]),
		] + args.x_scalers

	model_path = generate_config(args, create=x_train is not None)
	if args.verbose: print(f'Using model path {model_path}')

	preds = []
	for round_num in trange(args.n_rounds, disable=args.verbose or (args.n_rounds == 1) or args.silent):
		
		if args.seed is not None:
			np.random.seed(args.seed + round_num)

		x_scalers = list(args.x_scalers)
		y_scalers = list(args.y_scalers)

		if using_bagging and x_train is not None and args.n_rounds > 1:
			x_scalers, y_scalers, x_train, y_train, x_remain, y_remain = bagging_subset(args, x_train_orig, y_train_orig, x_scalers, y_scalers)
		else:
			x_remain = y_remain = None 

		scalerx = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in x_scalers])
		scalery = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in y_scalers])
		kwargs  = {
			'n_mix'     : args.n_mix, 
			'hidden'    : [args.n_hidden] * args.n_layers, 
			'lr'        : args.lr,
			'l2'        : args.l2,
			'n_iter'    : args.n_iter,
			'batch'     : args.batch,
			'avg_est'   : args.avg_est,
			'epsilon'   : args.epsilon,
			'threshold' : args.threshold,
			'scalerx'   : scalerx,
			'scalery'   : scalery,
			'model_path': model_path.joinpath(f'Round_{round_num}'),
			'no_load'   : args.no_load,
			'no_save'   : args.no_save,
			'seed'      : args.seed + round_num if args.seed is not None else None,
			'verbose'   : args.verbose,
		}

		model = MDN(**kwargs)
		x_remain = x_test
		y_remain = y_test
		model.fit(x_train, y_train, output_slices, args=args, x_valid=x_remain, y_valid=y_remain)

		if x_test is not None:
			partial = []
			chunk_size = args.batch * 100

			# To speed up the process and limit memory consumption, apply the trained model to the given test data in batches
			for i in trange(0, len(x_test), chunk_size, disable=not args.verbose):
				partial.append( model.predict(x_test[i:i+chunk_size]) )
			preds.append(np.vstack(partial))
			model.session.close()

			if args.verbose and y_test is not None:
				median = np.sum(preds, 0) if args.boosting else np.median(preds, 0)
				labels = [k + (f'{wavelengths[i]:.0f}' if (v.stop-v.start) > 1 else '') 
								for k,v in sorted(output_slices.items(), key=lambda pi: pi[1].start) 
								for i   in range(v.stop - v.start)][:y_test.shape[1]]
				for j, lbl in enumerate(labels):
					print( performance('%7s Median' % lbl, y_test[:, j], median[:, j]) )

				print(f'--- Done round {round_num} ---\n')

	return preds, model.output_slices


def apply_model(x_test, use_cmdline=True, **kwargs):
	''' 
	Apply a model (defined by kwargs 
	and default parameters) to x_test 
	'''
	args = get_args(kwargs, use_cmdline=use_cmdline)
	preds, idxs = get_estimates(args, x_test=x_test)
	return np.median(preds, 0), idxs


def train_model(x_train, y_train, **kwargs):
	args = get_args(kwargs)
	get_estimates(args, x_train, y_train, output_slices={'chl': slice(0,1)})


def image_estimates(*bands, sensor='', product_name='chl', **kwargs):
	''' 
	Takes any number of input bands (shaped [Height, Width]) and
	returns the products for that image, in the same shape. 
	Assumes the given bands are ordered by wavelength from least 
	to greatest, and are the same bands used to train the network.
	Supported products: {chl}
	'''
	args = get_args(kwargs, product=product_name, sensor=sensor)

	valid_products = ['chl']
	assert(sensor), (
		f'Must pass sensor name to image_estimates function')
	assert(sensor in SENSOR_LABEL), (
		f'Requested sensor {sensor} unknown. Must be one of: {list(SENSOR_LABEL.keys())}')
	assert(product_name in valid_products), (
		f'Requested product unknown. Must be one of {valid_products}')
	assert(all([bands[0].shape == b.shape for b in bands])), (
		f'Not all inputs have the same shape: {[b.shape for b in bands]}')
	assert(len(bands) == len(get_sensor_bands(sensor))), (
		f'Got {len(bands)} bands; expected {len(get_sensor_bands(sensor))} bands for sensor {sensor}')

	im_shape = bands[0].shape 
	im_data  = np.ma.vstack([np.ma.masked_invalid(b.flatten()) for b in bands]).T
	im_mask  = np.any(im_data.mask, axis=1)
	im_data  = im_data[~im_mask]
	pred,idx = get_estimates(args, x_test=im_data)
	products = np.median(pred, 0) 
	product  = np.atleast_2d( products[:, idx[product_name]] )
	est_mask = np.tile(im_mask[:,None], (1, product.shape[1]))
	est_data = np.ma.array(np.zeros(est_mask.shape), mask=est_mask, hard_mask=True)
	est_data.data[~im_mask] = product
	return [p.reshape(im_shape) for p in est_data.T]


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
		print('Generating estimates for %s data points' % len(x_test))
		preds, idxs = get_estimates(args, x_test=x_test)
		print(f'Min: {np.median(preds, 0).min(0)}')
		print(f'Max: {np.median(preds, 0).max(0)}')

		labels = get_labels(get_sensor_bands(args.sensor, args), idxs, preds[0].shape[1])
		preds  = np.append(np.array(labels)[None,:], np.median(preds, 0), 0)

		filename = filename.parent.joinpath(f'MDN_{filename.stem}.csv').as_posix()
		print(f'Saving estimates at location "{filename}"')
		np.savetxt(filename, preds.astype(str), delimiter=',', fmt='%s')

	elif args.plot_map:
		x_data, y_data, x_test, y_test, slices, (idxs, locs) = get_data(args)

		from .plot_map import draw_map
		import pandas as pd 
		
		name_lonlats = {
			'bunkei'      : (140, 36),
			'redriver'    : (104, 22),
			'ondrusek'    : (128, 35),
			'fremont'     : (-109, 43),
			'daniella'    : (-89, 42),
			'natascha'    : (10, 50),
			'moritz'      : (170, -44),
			'krista'      : (26, 58),
			'lakeerie'    : (-82.1, 41.9),
			'daniella'    : (-89, 42),
			'taihu'       : (120, 31),
			'vietnam'     : (107, 14),
			'ucsc'        : (-122.02, 36.96),
			'mississippi' : (-90.03, 32.44),
			'massbay'     : (-71, 42.34),
		}
		count_labels = lambda labels: '\n'.join([f'\t* {num:>4} | {name}' for name, num in zip(*np.unique(labels, return_counts=True))])

		dataset_lbls = np.zeros((len(x_data))).astype(str)
		full_lonlats = np.empty((len(x_data), 2))
		full_lonlats.fill(np.nan)

		for dataset in locs:
			if dataset not in dataset_lbls:
				path = Path(args.test_loc).joinpath(dataset)
				
				full_idxs = np.where(locs == dataset)[0]
				indv_idxs = idxs[full_idxs]
				dataset_lbls[full_idxs] = dataset
				print(f'{len(indv_idxs):>4} | {dataset}')

				if dataset == 'SeaBASS2':
					meta    = pd.read_csv(path.joinpath('meta.csv'), dtype=str).iloc[indv_idxs]
					lonlats = meta[['east_longitude', 'west_longitude', 'north_latitude', 'south_latitude']].apply(lambda v: v.apply(lambda v2: v2.split('||')[0]))
					assert(lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all().all())
					
					lonlats = lonlats.apply(lambda v: pd.to_numeric(v.apply(lambda v2: v2.split('::')[1].replace('[deg]','')), 'coerce'))
					assert(not lonlats.isna().any().any()), lonlats[lonlats.isna().any(axis=1)]
					full_lonlats[full_idxs] = lonlats[['east_longitude', 'north_latitude']].to_numpy()

				elif dataset == 'Sundar':
					location1 = np.loadtxt(path.joinpath('Dataset.csv'),  delimiter=',', dtype=str)[indv_idxs]
					location2 = np.loadtxt(path.joinpath('Location.csv'), delimiter=',', dtype=str)[indv_idxs]
					labels    = [(d+'_'+l).lower() for d,l in zip(location1, location2)]
					replace   = [('redriver', 'redriver'), ('ondrusek_korea', 'ondrusek'), ('fremontlake', 'fremont'), ('daniella', 'daniella'),
								('asian_lake', 'bunkei'), ('natascha', 'natascha'), ('moritz', 'moritz'), ('krista', 'krista'), ('lakeerie', 'lakeerie')]
					
					for curr, new in replace:
						labels = [v if curr not in v else new for v in labels]
					print( count_labels(labels) )
					
					curr_lonlats = []
					for name in labels:
						if name not in name_lonlats:
							raise Exception(f'Unknown location in {dataset}: {name}')	
						curr_lonlats.append(name_lonlats[name])
					full_lonlats[full_idxs] = np.array(curr_lonlats)

				elif path.joinpath('latlon.csv').exists():
					try:    lonlats = np.loadtxt(path.joinpath('latlon.csv'), delimiter=',')
					except: lonlats = np.loadtxt(path.joinpath('latlon.csv'), delimiter=',', skiprows=1)
					full_lonlats[full_idxs] = lonlats[indv_idxs][:,::-1].astype(np.float32)
				
				elif any([name in dataset.lower() for name in name_lonlats]):
					names = [name for name in name_lonlats if name in dataset.lower()]
					assert(len(names) == 1), f'Multiple possible names found for dataset {dataset}: {names}'
					full_lonlats[full_idxs] = name_lonlats[names[0]]

				else: raise Exception('No coordinates are set for {dataset}, and the coordinate file does not exist')

		assert('0' not in dataset_lbls), f'Error: {(dataset_lbls == "0").sum()} missing dataset indices'
		missing = np.any(np.isnan(full_lonlats), 1)
		if missing.any(): print(f'\nWarning: {missing.sum()} missing lon/lat coordinates:\n{count_labels(dataset_lbls[missing])}')
		ll = np.vstack(full_lonlats)
		
		# assert(0)
		# for a,b in zip(dataset_lbls, full_lonlats):
		# 	print(a,':',b)
		# assert(0)
		import matplotlib.pyplot as plt 

		import pandas as pd 
		d = pd.read_csv('parsed_WOD.csv'); n='WOD'
		# d = pd.read_csv('parsed_ndbc.csv'); n='NDBC'
		print(d.shape, d[['lon','lat']].to_numpy().shape)
		print(len(np.unique(d['lon'].to_numpy())), len(np.unique(d['lat'].to_numpy())))
		print(d['chl'].min(), d['chl'].max())
		draw_map(d[['lon','lat']].to_numpy(), color='r', edgecolor='grey', linewidth=0.1, s=8, world=True)
		plt.savefig(f'{n}_map.png', bbox_inches='tight', pad_inches=0.05, dpi=250)
		plt.show()
		assert(0)
		# Label each dataset
		if False:
			order = list(np.unique(dataset_lbls))
			full_lonlats = [full_lonlats[dataset_lbls == o] for o in order]
			draw_map(*full_lonlats, labels=order, s=8, world=True)
		else:
			# draw_map(full_lonlats, color='r',edgecolor='grey', linewidth=0.1, s=8, world=True)

			for lbl, k in [('na',(ll[:, 0] < -30) & (ll[:,1] > 10)), 
							('sa',(ll[:, 0] < -30) & (ll[:,1] < 10)),
							('aus',(ll[:, 0] > 50) & (ll[:,1] < 0)),
							('asia',(ll[:, 0] > 50) & (ll[:,1] > 0)),
							('eu',(ll[:, 0] < 50) & (ll[:,0] > -30) & (ll[:,1] > 0))]:
				print(lbl, k.sum())
				draw_map(ll[k], color='r',edgecolor='grey', linewidth=0.1, s=8, world=True)
				plt.show()
			assert(0)
		plt.savefig('insitu_map.png', bbox_inches='tight', pad_inches=0.05, dpi=250)
		plt.show()		


	# Train a model with partial data, and benchmark on remaining
	elif args.benchmark:
		# from xgboost import XGBRegressor as XGB 
		from collections import defaultdict as dd 
		from .benchmarks import run_benchmarks
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore')
			import matplotlib.pyplot as plt
			import matplotlib.gridspec as gridspec
			import matplotlib.ticker as ticker
			import matplotlib.patheffects as pe 
			import seaborn as sns 

		if args.test_set == 'paper':
			setattr(args, 'fix_tchl', True)
			setattr(args, 'seed', 1234)

		np.random.seed(args.seed)

		x_data, y_data, x_test, y_test, slices, locs = get_data(args)
		print('\n',x_data.shape, y_data.shape, np.array(locs).shape)
		# np.savetxt('data.csv', x_data, delimiter=',')
		# assert(0)
		data_idxs = np.arange(len(x_data))
		use_mdn   = True
		product   = args.product.split(',') if args.product != 'all' else ['chl']
		estimates = []

		for _ in range(1):

			if not args.use_sim:
				# Perform a single random split, and show scatterplots
				np.random.shuffle(data_idxs)
				n_train = 1000 if args.test_set == 'paper' else int(len(x_data)*0.5)#1000 if not args.use_sim else (len(x_data)-1)#int(0.50 * len(data_idxs))
				n_valid = int(0. * len(data_idxs))
				x_train = x_data[ data_idxs[:n_train] ]
				y_train = y_data[ data_idxs[:n_train] ]
				x_valid = x_data[ data_idxs[n_train:n_valid+n_train] ]
				y_valid = y_data[ data_idxs[n_train:n_valid+n_train] ]
				x_test  = x_data[ data_idxs[n_train+n_valid:] ]
				y_test  = y_data[ data_idxs[n_train+n_valid:] ]
			else:
				n_train = 'sim'
				x_train = x_data 
				y_train = y_data
				x_valid = np.zeros((0,x_data.shape[1]))
				y_valid = np.zeros((0,y_data.shape[1]))
			waves = get_sensor_bands(args.sensor, args)


			n_targets = y_test.shape[1]
			print('Min/Max Test X:', list(zip(np.nanmin(x_test, 0).round(2), np.nanmax(x_test, 0).round(2))))
			print('Min/Max Test Y:', list(zip(np.nanmin(y_test, 0).round(2), np.nanmax(y_test, 0).round(2))))
			if len(y_valid):
				print('Min/Max Valid:', list(zip(np.nanmin(y_valid, 0).round(2), np.nanmax(y_valid, 0).round(2))))
			if len(y_train):
				print('Min/Max Train X:', list(zip(np.nanmin(x_train, 0).round(2), np.nanmax(x_train, 0).round(2))))
				print('Min/Max Train Y:', list(zip(np.nanmin(y_train, 0).round(2)[:n_targets], np.nanmax(y_train, 0).round(2)[:n_targets])))
			print('Shapes:',x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape, y_test.shape)
			print(f'Train valid: {np.isfinite(y_train).sum(0)}')
			print(f'Test valid: {np.isfinite(y_test).sum(0)}')
			print(f'min/max wavelength: {waves[0]}, {waves[-1]}')
			print()

			if False:
				std  = np.std(y_data, 0)
				norm = np.mean(y_data, 0) / std 
				from .spectrum_rgb import get_spectrum_cmap 
				cmap = get_spectrum_cmap()
				bands = get_sensor_bands(args.sensor, args)
				colors = [cmap.to_rgba(nm) for nm in bands]
				fig, axes = plt.subplots(1, 3, sharey=False, figsize=(5*3,5*1))
				fig.add_subplot(111, frameon=False)
				plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
				plt.title(r'$a_{ph}\ Variability$', fontsize=18)
				for i, (name, val) in enumerate([('Mean', np.mean(y_data,0)), ('Std', std), ('Std/Mean', norm)]):
					ax = axes[i]
					ax.bar(bands, val, 5, color=colors)

					ax.set_ylabel(r'$\mathbf{%s}$' % name, fontsize=14)
					ax.set_xlabel(r'$\mathbf{Wavelength\ [nm]}$', fontsize=14)
					ax.tick_params(labelsize=14)
				plt.subplots_adjust(wspace=0.25)
				# plt.savefig(f'{args.sensor}_{args.product}_band_std.png')
				plt.show()
				assert(0)
			# < 3  : 487
			# 3-9  : 458
			# 9-23 : 473
			# > 23 : 444

			y_test_orig = y_test.copy()
			x_test_orig = x_test.copy()
			# for v in [
			# 	lambda y: y <= 0.94, 
			# 	lambda y: np.logical_and(y > 0.94, y <= 2.6), 
			# 	lambda y: np.logical_and(y > 2.6,  y <= 6.4), 
			# 	lambda y: np.logical_and(y > 6.4,  y <= 20), 
			# 	lambda y: np.logical_and(y > 20,   y <= 56), 
			# 	lambda y: np.logical_and(y > 56, y <= 154), 
			# 	lambda y: y > 154, 
			# ]:
			# v = v(y_test_orig.flatten())

			# v = np.logical_and(y_test.flatten() > 56, y_test.flatten() <= 154)
			# v = y_test.flatten() <= 0.94
			# x_test = x_test_orig[v]
			# y_test = y_test_orig[v]
			# print(y_test.mean(), np.std(y_test), np.median(y_test))
			# print(y_test.shape)
			# print(len(y_test), np.median(y_test))
			# assert(0)
			
			estimates, est_slice = get_estimates(args, x_train, y_train, x_test, y_test, slices)
			# test_est,  est_slice = estimate(args, x_test=x_test)
			# test_est  = np.median(test_est, 0)[:, est_slice[product]]
			estimates = np.median(estimates, 0)#[:, est_slice[product]]
			# estimates = np.array([np.nan] * len(x_train))


			# if True:
			# 	# idx = np.array([ 604,629,1020,411] ) 
			# 	from benchmarks import bench_iop
			# 	idx = np.array([108, 246])#np.argmax(y_test,0)])#np.arange(16,20)
			# 	print(idx)
			# 	np.savetxt('mdn_spectra.csv', estimates[idx], delimiter=',')
			# 	np.savetxt('hico_spectra.csv', y_test[idx], delimiter=',')
			# 	all_b = dict(zip(*bench_iop(args.sensor, x_test[:,:len(get_sensor_bands(args.sensor, args))], y_test,{product:slices[product]}, silent=True)[::-1]))
			# 	print(list(all_b.keys()))
			# 	print(get_sensor_bands(args.sensor, args))
			# 	np.savetxt('qaa_spectra.csv', np.array([all_b['QAA %s%s'%(product, n)][idx] for n in get_sensor_bands(args.sensor, args)]).T, delimiter=',')
			# 	np.savetxt('giop_spectra.csv', np.array([all_b['GIOP %s%s'%(product, n)][idx] for n in get_sensor_bands(args.sensor, args)]).T, delimiter=',')
			# 	assert(0)

			n_wvl  = len(get_sensor_bands(args.sensor, args))
			bench  = run_benchmarks(args, args.sensor, x_test[:,:n_wvl], y_test, {p:slices[p] for p in product}, silent=False, x_train=x_train, y_train=y_train, gridsearch=False)
			bdict  = bench
			labels = get_labels(get_sensor_bands(args.sensor, args), slices, y_test.shape[1])
			if 'aph' in product:
				color  = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:lightish green', 'xkcd:reddish', 'xkcd:bluish purple']
			else:
				color  = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:greyish blue', 'xkcd:goldenrod', 'xkcd:fresh green', 'xkcd:clay']#, 'xkcd:bluish purple', 'xkcd:reddish']

			y_log  = np.array([np.log10(y_test[:, slices[p]]) for p in product]).T 
			# bench  = [[(k,b[i:i+y_log.shape[1]]) for k,b in bench.items() for i in range(0, len(bench), y_log.shape[1])]
			
			# from benchmarks import bench_opt
			# print(x_train.shape)
			# bdict.update(dict(zip(*bench_opt(args.sensor, x_train[:,:n_wvl], x_test[:,:n_wvl], y_train, y_test, {product:slices[product]})[::-1])))
			
			print()
			for p in product:
				for lbl, y1, y2 in zip(labels[slices[p]], y_test.T[slices[p]], estimates.T[slices[p]]):
					print( performance(f'MDN {lbl}', y1, y2) ) 
			print()

			if y_test.shape[1] == 1:
				mwrs = 'MWRs\n'
				for p in product:
					for k in bdict:
						# if k in ['OC3', 'Blend']:
						mwrs += '%s: %.3f   \n' % (k, mwr(y_test, estimates[:,slices[p]], bdict[k]))
				print(mwrs,'\n')
			# input('next?')

			# assert(0)
			# assert(0)
			# from QAA import find as find_wavelength
			# ratio = x_test[:,find_wavelength(708, np.array(wavelengths['S2B']))] / x_test[:,find_wavelength(665, np.array(wavelengths['S2B']))]
			# i, chl = zip(*sorted(enumerate(y_test.flatten()), key=lambda k:k[1]))
			# print(ratio.shape, x_test.shape)
			# plt.plot([ratio[ii] for ii in i], color='orange', label='Ratio')
			# plt.axhline(0.75, color='k', ls='--')
			# plt.axhline(1.15, color='k', ls='--')
			# plt.ylabel('705nm / 665nm')
			# plt.ylim(0, 5)
			# plt.legend()
			# plt.twinx()
			# plt.plot(chl, label='Chl')
			# plt.legend()
			# plt.ylabel('Chl')
			# plt.show()
			# assert(0)

			# rtrial_ests.append([estimates, dict(zip(*run_benchmarks(args.sensor, x_test[:,:n_wvl], y_test, {product:slices[product]}, silent=True)[::-1])), bench])
			# rtrial_true.append(y_test[:,slices[product]])
			if True:
				# assert(0)
				plt.rc('text', usetex=True)
				plt.rcParams['mathtext.default']='regular'

				#fmt  = ticker.FormatStrFormatter(r'$10^{%i}$')
				fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

				seaplts = []
				n_plots = 3
				n_plots = min(n_plots, 1+len(bench))
				n_target= len(product)

				valid_samples = np.array([True] * len(x_test))
				valid_plot = [True] * len(labels)
				if len(labels) > 3 and 'chl' not in product:
					if False:
						from .spectrum_rgb import get_spectrum_cmap 
						rsr = np.loadtxt(f'/media/brandon/NASA/Data/Rsr/{args.sensor}_rsr.csv', delimiter=',')
						cmap = get_spectrum_cmap()
						bands = get_sensor_bands(args.sensor, args)
						print(bands, slices, product)
						colors = [cmap.to_rgba(nm) for nm in bands]
						metrics = [mae, rmsle, mape]
						fig, axes = plt.subplots(1, len(metrics), sharey=False, figsize=(9*n_plots,5*n_target))
						product = 'aph'

						for i,m in enumerate(metrics):
							ax = axes[i]
							vy = np.all(np.isfinite(y_test[:, slices[product]]), 1)
							errs = [m(y, e) for y, e in zip(y_test[vy, slices[product]].T, estimates[vy, slices[product]].T)]
							# std  = np.std(y_test, 0) #/ np.mean(y_test, 0)
							# errs = np.array(errs) / std

							for j,(b,e) in enumerate(zip(bands, errs)):
								# r = rsr[1:,list(rsr[0]).index(b)]
								# r /= r.max()
								# r *= e 

								# if m.__name__ == 'MAPE':
								# 	e *= 100
								if False: # use the actual RSR function to plot each band
									ax.plot(rsr[1:,0], r, color=colors[j])
									ax.fill_between(rsr[1:,0], r, 0, color=colors[j], alpha=0.5)
									ax.axhline([0], color='k')
								else: 
									# ax.bar([b], [e], (r > (e*0.99)).sum(), color=colors[j])
									ax.bar([b], [e], 5, color=colors[j])
							# ax.bar(bands, errs, color=colors)
							ax.set_xlim((bands.min()-10, bands.max()+10))
							if m.__name__ == 'MAE':
								ax.set_ylim((1, ax.get_ylim()[1]))
							if m.__name__ == 'MAPE':
								ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))#FuncFormatter(lambda i, _: '%i\%' % round(i*100)))
								# ax.set_yticklabels(['%i\%' % round(v*100) for v in ax.get_yticks()])
							ax.set_ylabel(r'$\mathbf{%s}$' % m.__name__, fontsize=14)
							ax.set_xlabel(r'$\mathbf{Wavelength\ [nm]}$', fontsize=14)
							ax.tick_params(labelsize=14)
						plt.subplots_adjust(wspace=0.25)
						plt.show()
						assert(0)

					target     = [443, 590, 620, 670] if product == 'a*ph' else [443, 530,] if product == 'aph' else [443,482,561,655]
					waves      = np.array(get_sensor_bands(args.sensor, args))
					valid_wvls = [waves[find_wavelength(w, waves)] for w in target]
					valid_plot = [w in valid_wvls for w in waves]
					n_target   = len(target)

					if False:
						valid_samples = np.all([np.all([np.logical_and(b[0] > 0, np.isfinite(b[0])) for b in benches if any([lbl in b[1] for lbl in ['QAA', 'GIOP']])], 0) for i,benches in enumerate(zip(*bench)) if valid_plot[i]], 0)
						f, axes = plt.subplots(2,2, figsize=(10,10))

						# qa = np.array([v[1] for v in sorted([(k,bdict[k]) for k in bdict if 'QAA' in k], key=lambda z:int(z[0].split('aph')[1]))]).T[~valid_samples].T
						# ga = np.array([v[1] for v in sorted([(k,bdict[k]) for k in bdict if 'GIOP' in k], key=lambda z:int(z[0].split('aph')[1]))]).T[~valid_samples].T
						qa = np.array([v[1] for v in sorted([(k,bdict[k]) for k in bdict if 'QAA' in k], key=lambda z:int(z[0].split('aph')[1]))]).T
						ga = np.array([v[1] for v in sorted([(k,bdict[k]) for k in bdict if 'GIOP' in k], key=lambda z:int(z[0].split('aph')[1]))]).T
						vq = np.any(qa <= 0, 1)
						vg = np.any(ga <= 0, 1)
						qa = qa[vq].T 
						ga = ga[vg].T

						axes[1][0].plot(waves, qa, color='b', alpha=0.1)
						axes[1][1].plot(waves, ga, color='orange', alpha=0.2)
						# axes[1][0].plot(waves, qa[:,np.all(qa < 2, 0)], color='b', alpha=0.2)
						# axes[1][1].plot(waves, ga[:,np.all(ga < 2, 0)], color='orange', alpha=0.2)
						axes[0][0].plot(waves, x_test[vq].T, color='r', alpha=0.1)#, ls='--')
						axes[0][1].plot(waves, x_test[vg].T, color='r', alpha=0.2)#, ls='--')
						axes[0][0].set_xticklabels([])
						axes[0][1].set_xticklabels([])
						# axes[0][1].set_yticklabels([])
						axes[0][0].set_ylim((0, 0.05))
						axes[0][1].set_ylim((0, 0.05))
						axes[1][0].set_ylim((-0.5, 1))
						axes[1][1].set_ylim((-0.2, 0))
						# axes[1][1].set_ylim((-0.2, axes[1][1].get_ylim()[1]))
						# axes[1][2].set_yscale('log')

						# axes[0][0].set_title('aph')
						axes[1][0].set_ylabel(r'$\mathbf{a_{ph}\ [1/m]}$', fontsize=24)
						axes[0][0].set_ylabel(r'$\mathbf{R_{rs}\ [1/sr]}$', fontsize=24)
						axes[1][0].set_xlabel(r'$\mathbf{Wavelength\ [nm]}$', fontsize=24)
						axes[1][1].set_xlabel(r'$\mathbf{Wavelength\ [nm]}$', fontsize=24)						
						axes[0][0].set_title(r'$\mathbf{QAA}$'+ f'\nN={len(qa.T)}', fontsize=24)
						axes[0][1].set_title(r'$\mathbf{GIOP}$'+ f'\nN={len(ga.T)}', fontsize=24)
						
						axes[0][0].tick_params(labelsize=18)
						axes[0][1].tick_params(labelsize=18)
						axes[1][0].tick_params(labelsize=18)
						axes[1][1].tick_params(labelsize=18)
						plt.subplots_adjust(hspace=0.1, wspace=0.3)
						plt.savefig('negative_aph_individual.png')
						plt.show()
						# assert(0)
						# assert(0)
					valid_samples = np.ones_like(valid_samples).astype(np.bool)
					# x_data = np.append(x_train, x_test[valid_samples], axis=0)
					# y_data = np.append(y_train, y_test[valid_samples], axis=0)
					# x_data = np.append(x_data, y_data, axis=1)
					# # x_data = np.append(np.array(locs)[:, None], x_data, axis=1)
					# x_data = np.append(np.array([f'Rrs{b}' for b in get_sensor_bands(args.sensor, args)] + [f'{args.product}{b}' for b in get_sensor_bands(args.sensor, args)])[None,:], x_data, axis=0)
					# np.savetxt(f'{args.sensor}_{args.product}.csv',x_data, delimiter=',', fmt='%s')
					# assert(0), x_data.shape

					print(valid_samples.shape)
				labels = [(p,label) for label in labels for p in product if p in label]
				print('Plotting labels:', [l for i,l in enumerate(labels) if valid_plot[i]])

				if not len(bench):
					bench = [[None] * y_log.shape[1]]
				
				# fig, axes = plt.subplots(2, 2, sharey=False, figsize=(8,7))
				bench_order = None
				if n_plots > 3 or 'chl' in product:
					# bench_order = ['Smith_Blend', 'OC3', 'Mishra_NDCI', 'Gons_2band', 'Gilerson_2band']
					bench_order = ['OC3', 'XGB', 'SVM', 'MLP', 'KNN']
					bench_order = [b for b in bench_order if b in bdict]
					n_plots = len(bench_order) + 1
					fig, axes = plt.subplots(2, (n_plots+1)//2, sharey=False, figsize=(5*((n_plots+1)//2),10))
					# fig, axes = plt.subplots(n_target, n_plots, sharey=False, figsize=(n_plots*5,n_target*5+2))

				else:
					bench_order = ['QAA']#, 'GIOP']
					# bench_order = [b for b in bench_order if b in bdict]
					n_plots = len(bench_order) + 1
					fig, axes = plt.subplots(n_target, n_plots, sharey=False, figsize=(5*n_plots,5*n_target))

				print('Order:', bench_order)
				print('n plots:', n_plots)
				if (n_target > 1 and n_plots > 1) or n_plots > 3: 
					axes = [ax for axs in axes for ax in axs]
				print('n axes:', len(axes))
				print('n target:',n_target)
				fig.add_subplot(111, frameon=False)
				plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
				# axes[0][1].axis('off')
				curr_idx = 0
				# benches = [{k:v[:,i] for k,v in bench.items()} for i in range(slices[product].start, slices[product].stop)]
				y_test = np.array([y_test[:, slices[p]] for p in product])
				estimates = np.array([estimates[:, slices[p]] for p in product])
				for plt_idx, (title, y_true, y_orig, y_model) in enumerate(zip(labels, y_log.T, y_test, estimates)):
					product, title = title 

					if not valid_plot[plt_idx]:
						continue 

					# fig = plt.figure(figsize=(2*4,7))
					y_true = y_true[valid_samples]
					y_orig = y_orig[valid_samples]
					y_model = y_model[valid_samples]

					y_true = y_true.flatten()
					method_lbls = []
					for i in range(n_plots):

						# ax = plt.subplot2grid((1, n_plots), (0, i))
						# ax = plt.subplot2grid((2, 2), (0 if i == 0 else 1, i if i == 0 else i-1))
						# ax = axes[0 if i == 0 else 1][i if i == 0 else i-1]
						ax = axes[curr_idx]
						curr_idx += 1

						s_kws = {'alpha': 0.4, 'color': color[i]}
						l_kws = {'color': color[i], 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}

						if i:
							if bench_order is not None:
								order = bench_order[i-1]
								print(i, 'Fetching', order, title)
								label = [b for b in bench if order in b]
								if len(label) > 1:
									label = [b for b in label if title in b]
									assert(len(label) == 1), [label, order, title] 
								label   = label[0]
								y_bench = bench[label]
								# y_bench, label = [b for b in benches if order in b[1]][0]
								label = label.replace('Mishra_','')

							else:
								OC_lbl = 'Blend'#'OC6' if args.sensor != 'OLCI' else 'OC4'
								if i == 1 and any([OC_lbl in b[1] for b in benches]):
									y_bench, label = [b for b in benches if OC_lbl in b[1]][0]
								elif i == 2 and any(['Blend' in b[1] for b in benches]):
									y_bench, label = [b for b in benches if 'Blend' in b[1]][0]
								elif i == 3 and any(['Mishra_NDCI' in b[1] for b in benches]):
									y_bench, label = [b for b in benches if 'Mishra_NDCI' in b[1]][0]
									label = 'NDCI'
								else:
									y_bench, label = benches[i-1]
							y_bench = y_bench[valid_samples].flatten()
							label = label.replace('Smith_','')#.replace('GIOP', 'GIOP-DC')
							y_est = np.log10(y_bench).flatten()
							y_est_orig = y_bench.copy()

						else:
							[i.set_linewidth(5) for i in ax.spines.values()]
							y_model = y_model.flatten()
							y_est = np.log10(y_model)
							y_est_orig = y_model.copy()
							label = 'MDN_{A}'
			
						method_lbls.append(label.split(' ')[0])
						prod_lbl = product.replace('chl', 'Chla').replace('ph', '_{ph}').replace('apg','a_{pg}').replace('ap','a_{p}')
						prod_lbl = prod_lbl.replace('ad', 'a_{d}').replace('ag', 'a_{cdom}').replace('*', '^{*}').replace('tss', 'SPM')
						if (hasattr(args, 'fix_tchl') and not args.fix_tchl) or (hasattr(args, 'keep_tchl') and args.keep_tchl):
							prod_lbl = prod_lbl.replace('Chla', 'TChla')
						unit = r'[mg/m^{3}]' if product=='chl' else r'[g/m^{3}]' if product == 'tss' else r'[1/m]'

						if curr_idx == 1: 
							plt.ylabel(fr'$\mathbf{{Modeled\ {prod_lbl}\ '+unit + r'}$', fontsize=20, fontweight='bold', labelpad=10)
							# plt.xlabel(r'$\mathbf{%s}$' % (r'In\ situ\ '+fr'{prod_lbl}\ '+unit) +'\n'+ r'$\small{\mathit{N\small{=}}%s}$'%len(y_true), fontsize=20, fontweight='bold', labelpad=10)
							plt.xlabel(r'$\mathbf{%s}$' % (r'Measured\ '+fr'{prod_lbl}\ '+unit), fontsize=20, fontweight='bold', labelpad=10)
							plot_title = get_sensor_label(args.sensor).replace('-',' ').replace(' ', '\ ')
							# plt.title(r'$\mathbf{\underline{\large{In\ Situ}}}$', fontsize=24, y=1.05)
							# plot_title = 'In\ Situ'
							plot_title = 'Type\ A'
							plt.title(r'$\mathbf{\underline{\large{%s}}}$'%plot_title + '\n' + r'$\small{\mathit{N\small{=}}%s}$'%len(y_true), fontsize=24, y=1.06)

						if product not in ['chl', 'tss']:
							wvl = int(get_sensor_bands(args.sensor, args)[plt_idx])
							prod_lbl += f'(\small{{{wvl}nm}})'

							if i == (n_plots-1):
								ax2 = ax.twinx()
								ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
								ax2.grid(False)
								ax2.set_ylabel(fr'$\mathbf{{{prod_lbl}}}$', fontsize=20)

						minv = -4 if product in ['aph', 'a*ph', 'bb_p'] else -2 #int(y_true.min()) - 2
						maxv = int(np.nanmax(y_true)) + 1 if product != 'aph' else 1#+ (1 if product in ['aph'] else 0)
						loc  = ticker.LinearLocator(numticks=maxv-minv+1)

						ax.set_ylim((minv, maxv))
						ax.set_xlim((minv, maxv))
						ax.xaxis.set_major_locator(loc)
						ax.yaxis.set_major_locator(loc)
						ax.xaxis.set_major_formatter(fmt)
					
						if (n_targets > 1 and curr_idx <= (len(axes) - n_plots)) or (n_plots > 3 and curr_idx <= (len(axes) - 3)):
							ax.set_xticklabels([])

						if i and not (i == (n_plots//2) and n_plots > 3):
							ax.set_yticklabels([])
						else:
							ax.yaxis.set_major_formatter(fmt)

						valid = np.logical_and(np.isfinite(y_true), np.isfinite(y_est))

						sns.regplot(y_true[valid], y_est[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws)#,label=r'$\mathrm{%s}$'%label.split(' ')[0])
						kde = sns.kdeplot(y_true[valid], y_est[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color[(i+1)%n_plots])
						kde.collections[2].set_alpha(0)

						if len(valid.flatten()) != valid.sum():
							ax.scatter(y_true[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (~valid).sum())
							ax.legend(loc='lower right', fontsize=16, prop={'weight':'bold'})

						add_identity(ax, ls='--', color='k', zorder=20)
						add_stats_box(ax, y_orig, y_est_orig)

						if curr_idx <= n_plots:
							ax.set_title(r'$\mathbf{\large{%s}}$' % label.split(' ')[0].replace('_2band','\ 2\ Band'), fontsize=18)

						ax.tick_params(labelsize=18)
						ax.grid('on', alpha=0.3)

				plt.tight_layout()
				plt.savefig(f'scatters/{args.product}_{args.sensor}_scatter_{n_train}train.png', dpi=100, bbox_inches='tight', pad_inches=0.1,)
				plt.show()

	# Otherwise, train a model with all data (if not already existing)
	else:
		x_data, y_data, x_test, y_test, slices = data = get_data(args)[:-1]
		get_estimates(args, *data)

