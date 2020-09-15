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
from .meta  import get_sensor_bands, get_sensor_label, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, find_wavelength, generate_config, using_feature, split_data
from .metrics import performance, rmse, rmsle, leqznan, mape, r_squared, slope, nrmse, mwr, bias, mae, mdsa
from .plot_utils import add_identity, add_stats_box
from .benchmarks import print_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, LogTransformer, NegLogTransformer, RatioTransformer, BaggingColumnTransformer


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
			model.session.close()

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
	elif anc: sensor = sensor.replace('S2B', 'MSI')

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

	elif args.plot_map:
		x_data, y_data, slices, locs = get_data(args)
		locs, idxs = locs.T 
		idxs = idxs.astype(int)

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

			'rio_de_la_plata' : (-58.4, -34.56), # Rio de la Plata
			'scheldt_river' : (4.4, 51.23), # Scheldt River
			'gironde_river' : (-0.72, 45.2), # Gironde River
			'hubert': (108, 15),
			'cedric': (np.nan, np.nan),

 		}
		count_labels = lambda labels: '\n'.join([f'\t* {num:>4} | {name}' for name, num in zip(*np.unique(labels, return_counts=True))])

		dataset_lbls = np.zeros((len(x_data))).astype(str)
		full_lonlats = np.empty((len(x_data), 2))
		full_lonlats.fill(np.nan)

		for dataset in locs:
			if dataset not in dataset_lbls:
				path = Path(args.data_loc).joinpath(dataset)
				
				full_idxs = np.where(locs == dataset)[0]
				indv_idxs = np.array(idxs)[full_idxs]
				dataset_lbls[full_idxs] = dataset
				print(f'{len(indv_idxs):>4} | {dataset}')

				if dataset == 'SeaBASS2':
					meta    = pd.read_csv(path.joinpath('meta.csv'), dtype=str).iloc[indv_idxs]
					lonlats = meta[['east_longitude', 'west_longitude', 'north_latitude', 'south_latitude']].apply(lambda v: v.apply(lambda v2: v2.split('||')[0]))
					# assert(lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all().all()), lonlats[~lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all(1)]
					
					lonlats = lonlats.apply(lambda v: pd.to_numeric(v.apply(lambda v2: v2.split('::')[1].replace('[deg]','')), 'coerce'))
					assert(not lonlats.isna().any().any()), lonlats[lonlats.isna().any(axis=1)]
					full_lonlats[full_idxs] = lonlats[['east_longitude', 'north_latitude']].to_numpy()

				elif dataset == 'Sundar':
					location1 = np.loadtxt(path.joinpath('dataset.csv'),  delimiter=',', dtype=str)[indv_idxs]
					location2 = np.loadtxt(path.joinpath('location.csv'), delimiter=',', dtype=str)[indv_idxs]
					labels    = [(d+'_'+l).lower() for d,l in zip(location1, location2)]
					replace   = [('redriver', 'redriver'), ('ondrusek_korea', 'ondrusek'), ('fremontlake', 'fremont'), ('daniella', 'daniella'),
								('asian_lake', 'bunkei'), ('natascha', 'natascha'), ('moritz', 'moritz'), ('krista', 'krista'), ('lakeerie', 'lakeerie'),
								('seaswir_1', 'scheldt_river'), ('seaswir_2', 'gironde_river'), ('seaswir', 'rio_de_la_plata'), ('hubert', 'hubert'), ('cedric', 'cedric')]
					
					for curr, new in replace:
						labels = [v if curr not in v else new for v in labels]
					print( count_labels(labels) )
					assert(0)

					curr_lonlats = []
					for name in labels:
						if name not in name_lonlats:
							if 'seaswir' in name: continue
							raise Exception(f'Unknown location in {dataset}: {name}')	
						curr_lonlats.append(name_lonlats[name])
					full_lonlats[full_idxs] = np.array(curr_lonlats)

				elif path.joinpath('latlon.csv').exists():
					print(path)
					try:    lonlats = np.loadtxt(path.joinpath('latlon.csv'), delimiter=',')
					except: lonlats = np.loadtxt(path.joinpath('latlon.csv'), delimiter=',', skiprows=1)
					full_lonlats[full_idxs] = lonlats[indv_idxs][:,::-1].astype(np.float32)
				
				elif any([name in dataset.lower() for name in name_lonlats]):
					names = [name for name in name_lonlats if name in dataset.lower()]
					assert(len(names) == 1), f'Multiple possible names found for dataset {dataset}: {names}'
					full_lonlats[full_idxs] = name_lonlats[names[0]]

				else: raise Exception(f'No coordinates are set for {dataset}, and the coordinate file does not exist')

		assert('0' not in dataset_lbls), f'Error: {(dataset_lbls == "0").sum()} missing dataset indices'
		missing = np.any(np.isnan(full_lonlats), 1)
		if missing.any(): print(f'\nWarning: {missing.sum()} missing lon/lat coordinates:\n{count_labels(dataset_lbls[missing])}')
		ll = np.vstack(full_lonlats)
		
		AERONET = [
			[33.104,130.272],
			[40.717, 1.358],
			[43.045, 28.193],
			[38.108,-122.056],
			[58.594, 17.467],
			[59.949, 24.926],
			[35.611,140.023],
			[41.826, -83.194],
			[26.902, -80.789],
			[40.955, -73.342],
			[-18.520,146.386],
			[41.325, -70.567],
			[58.755, 13.152],
			[44.546, 29.447],
			[37.423,124.738],
			[44.596, -87.951],
			[33.564,-118.118],
			[45.314, 12.508],
			[28.867, -90.483],
			[51.362, 3.120],
		]

		Vaal = [
			[28.115647, -26.889183],
			[28.121753, -26.889084],
			[28.138747, -26.888203],
			[28.148568, -26.877874],
			[28.155088, -26.868202],
			[28.156423, -26.866618],
			[28.154825, -26.828902],
			[28.166677, -26.868113],
			[28.157918, -26.8869],
			[28.128792, -26.904233],
			[28.137223, -26.892167],
			[28.189773, -26.993363],
			[28.218161, -26.986757],
			[28.231925, -26.97359],
			[28.193542, -26.9423],
			[28.199676, -26.920006],
			[28.166869, -26.888858],
			[28.14583, -26.89515],
			[28.14128, -26.908375],
			[28.110508, -26.941792],
			[28.240201, -26.881554],
			[28.227307, -26.891153],
			[28.229853, -26.919571],
		]
		india = {
			'Chennai': [80.32167, 13.13725], 
			'Harbour': [80.302, 13.12762], 
			'Muttukaadu': [80.23333, 12.81667], 
			'Point Calimere': [79.97333, 10.14833],
		}


		# for a,b in zip(dataset_lbls, full_lonlats):
		# 	print(a,':',b)
		# assert(0)
		import matplotlib.pyplot as plt 

		import pandas as pd 
		# d = pd.read_csv('parsed_WOD.csv'); n='WOD'
		# d = pd.read_csv('parsed_ndbc.csv'); n='NDBC'
		# print(d.shape, d[['lon','lat']].to_numpy().shape)
		# print(len(np.unique(d['lon'].to_numpy())), len(np.unique(d['lat'].to_numpy())))
		# print(d['chl'].min(), d['chl'].max())
		# draw_map([d[['lon','lat']].to_numpy(), AERONET], labels=['Training', 'AERONET-OC'], color='r', edgecolor='grey', linewidth=0.1, s=8, world=True)
		
		# valid_data = np.array(AERONET)[:,::-1]
		# ll = np.append(ll, Vaal, 0)
		# ll = np.append(ll, list(india.values()), 0)
		# print(ll.shape, valid_data.shape)
		# draw_map(ll, valid_data, labels=['Development data', 'Validation data'],  edgecolor='grey', linewidth=0.3, s=20, world=True, us=False)
		# plt.savefig(f'map.png', bbox_inches='tight', pad_inches=0.05, dpi=250)
		# plt.show()
		# assert(0)

		# Label each dataset
		if True:
			order = list(np.unique(dataset_lbls))
			order = [o for o in order if o not in ['Bunkei_a', 'Taihu3']]
			# order = ['SeaBASS', 'Other Sources']
			full_lonlats = [full_lonlats[dataset_lbls == o] for o in order]
			# full_lonlats = [full_lonlats[dataset_lbls == 'SeaBASS2'], full_lonlats[dataset_lbls != 'SeaBASS2']]
			order = [o.replace('2','').replace('_a','').replace('Caren', 'Lake Erie').replace('Fremont','Nebraska Lakes').replace('Gurlin3', 'Wisconsin Lakes').replace('Mississippi', 'Mississippi Ponds').replace('Indiana', 'Indiana Lakes')+ f' ({len(full_lonlats[i])})' for i,o in enumerate(order)]
			draw_map(*full_lonlats, labels=order, s=30, world=False, us=True, gray=True, edgecolor='grey', linewidth=0.2)
		else:
			# draw_map(full_lonlats, color='r',edgecolor='grey', linewidth=0.1, s=8, world=True)
			# print(len(ll), 'total')
			# for lbl, k in [('na',(ll[:, 0] < -30) & (ll[:,1] > 10)), 
			# 				('sa',(ll[:, 0] < -30) & (ll[:,1] < 10)),
			# 				('aus',(ll[:, 0] > 50) & (ll[:,1] < 0)),
			# 				('asia',(ll[:, 0] > 50) & (ll[:,1] > 0)),
			# 				('india',(ll[:, 0] > 50) & (ll[:,0] < 90) & (ll[:,1] > 0)),
			# 				('eu',(ll[:, 0] < 50) & (ll[:,0] > -30) & (ll[:,1] > 0))]:
			for lbl, k in [('erie', ((ll[:, 0] < -79) & (ll[:, 0] > -84) & (ll[:, 1] < 43) & (ll[:, 1] > 38)))]:
				print(lbl, k.sum())
				print(np.unique(np.array(dataset_lbls)[k]))
				draw_map(ll[k], color='r',edgecolor='grey', linewidth=0.1, s=8, world=True, us=False)
				plt.show()
			assert(0)
		plt.savefig(f'insitu_{args.sensor}_{args.product}_map.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
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

		if args.dataset == 'sentinel_paper':
			setattr(args, 'fix_tchl', True)
			setattr(args, 'seed', 1234)

		np.random.seed(args.seed)

		# s, args.sat_bands = args.sat_bands, False
		# a, args.align = args.align, 'HICO-sat'
		x_data_bench, y_data_bench, slices_bench, locs_bench = get_data(args)
		print(x_data_bench.shape)

		# args.align = 'HICO'
		# args.sat_bands = s
		x_data, y_data, slices, locs = get_data(args)

		print('\n',x_data.shape, y_data.shape, np.array(locs).shape)
		waves = get_sensor_bands(args.sensor, args)

		# for y1, y2 in zip(y_data, y_orig):
		# 	plt.plot(bands, y1[slices[args.product]], label='new')
		# 	plt.plot(bands, y2[slices[args.product]], label='orig')
		# 	plt.legend()
		# 	plt.show()

		# print(np.sum(np.isfinite(y_data) & (y_data > 0), 0))
		# print(slices)
		# sns.set()
		# plt.rc('text', usetex=True)
		# plt.rcParams['mathtext.default']='regular'
		# plt.rcParams['axes.labelweight'] = 'bold'
		# plt.rc('font',weight='bold')
		# Rrs665 = x_data[:,4]
		# Rrs665 = Rrs665[np.isfinite(Rrs665) & (Rrs665 > 0)]
		# print(len(Rrs665))
		# sns.distplot(np.log10(Rrs665))#, hist_kws={'log':True})
		# fmt  = ticker.FuncFormatter(lambda i, _: r'$\mathbf{10}$\textsuperscript{%i}'%i)
		# loc  = ticker.FixedLocator([-4, -3, -2, -1])
		# plt.gca().xaxis.set_major_formatter(fmt)
		# plt.gca().xaxis.set_major_locator(loc)
		# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda i,_: r'$\mathbf{%.1f}$' % i))
		# # plt.xscale('log')
		# plt.title(r'$\mathbf{R_{rs}(665nm)}$', fontsize=32)
		# plt.xlabel(r'$\mathbf{R_{rs} [1/sr]}$', fontsize=30)
		# plt.ylabel(r'$\mathbf{Normalized\ Frequency}$', fontsize=22)
		# plt.gca().tick_params(labelsize=18)
		# plt.gca().tick_params(axis='x', labelsize=24)
		# plt.xlim((-4.5, -0.5))
		# plt.setp(plt.gca().get_yticklabels(), fontweight="bold")
		# plt.setp(plt.gca().get_xticklabels(), fontweight="bold")
		# ax = plt.gca()
		# plt.tight_layout()
		# plt.savefig('Rrs665.png', dpi=150, **{'bbox_inches' : 'tight',
		# 'pad_inches'  : 0.1})
		# plt.show()
		# assert(0)

		# # AC exercise data
		# python3 -m MDN --benchmark --sensor OLI-nan --product all
		# valid  = np.any(np.isfinite(x_data), 1)
		# x_data = x_data[valid].astype(str)
		# y_data = y_data[valid].astype(str)
		# locs   = np.array(locs).T[valid].astype(str)
		# wvls = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
		# lbls = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
		# data = np.append([wvls], x_data.astype(str), 0)
		# data_full = np.append(np.append(locs, x_data, 1), y_data, 1)
		# data_full = np.append([['index', 'dataset']+wvls+lbls], data_full, 0)
		# # np.savetxt('hico_data.csv', data, delimiter=',', fmt='%s')
		# np.savetxt('hico_data_full.csv', data_full, delimiter=',', fmt='%s')
		# assert(0)

		data_idxs = np.arange(len(x_data))
		use_mdn   = True
		product   = args.product.split(',') if args.product != 'all' else ['chl', 'tss','cdom']
		estimates = []

		for _ in range(1):
			n_train = 0.5 if args.dataset != 'sentinel_paper' else 1000
			(x_train, y_train), (x_test, y_test) = split_data(x_data, y_data, n_train=n_train, seed=args.seed)
			(x_train_bench, y_train_bench), (x_test_bench, y_test_bench) = split_data(x_data_bench, y_data_bench, n_train=n_train, seed=args.seed)

			y_train = y_train[..., :1]
			y_test  = y_test[..., :1]
			# if not args.use_sim:
			# 	# Perform a single random split, and show scatterplots
			# 	np.random.shuffle(data_idxs)
			# 	n_train = 1000 if args.dataset == 'sentinel_paper' else int(len(x_data)*0.5)#1000 if not args.use_sim else (len(x_data)-1)#int(0.50 * len(data_idxs))
			# 	n_valid = int(0. * len(data_idxs))
			# 	x_train = x_data[ data_idxs[:n_train] ]
			# 	y_train = y_data[ data_idxs[:n_train] ]
			# 	x_valid = x_data[ data_idxs[n_train:n_valid+n_train] ]
			# 	y_valid = y_data[ data_idxs[n_train:n_valid+n_train] ]
			# 	x_test  = x_data[ data_idxs[n_train+n_valid:] ]
			# 	y_test  = y_data[ data_idxs[n_train+n_valid:] ]
			# else:
			# 	n_train = 'sim'
			# 	x_train = x_data 
			# 	y_train = y_data
			# 	x_valid = np.zeros((0,x_data.shape[1]))
			# 	y_valid = np.zeros((0,y_data.shape[1]))



			n_targets = y_test.shape[1]
			print('Min/Max Test X:', list(zip(np.nanmin(x_test, 0).round(2), np.nanmax(x_test, 0).round(2))))
			print('Min/Max Test Y:', list(zip(np.nanmin(y_test, 0).round(2), np.nanmax(y_test, 0).round(2))))
			if len(y_train):
				print('Min/Max Train X:', list(zip(np.nanmin(x_train, 0).round(2), np.nanmax(x_train, 0).round(2))))
				print('Min/Max Train Y:', list(zip(np.nanmin(y_train, 0).round(2)[:n_targets], np.nanmax(y_train, 0).round(2)[:n_targets])))
			print('Shapes:',x_train.shape, x_test.shape, y_train.shape, y_test.shape)
			print(f'Train valid: {np.isfinite(y_train).sum(0)}')
			print(f'Test valid: {np.isfinite(y_test).sum(0)}')
			print(f'min/max wavelength: {waves[0]}, {waves[-1]}')
			print()


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
			# args.no_load = False
			# for p2 in [1200, 1500, 1750, 2000]:
			# 	print('n_hidden:',p2)
			# # 	for p1 in [1,2,3,4,5,6,7,8,9]:
			# 	args.n_hidden = p2
			# # 		args.n_layers = p1 
			# 	estimates, est_slice = get_estimates(args, x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), slices)
			# 	# # test_est,  est_slice = estimate(args, x_test=x_test)
			# 	# # test_est  = np.median(test_est, 0)[:, est_slice[product]]
			# 	estimates = np.median(estimates, 0)#[:, est_slice[product]]
			# # estimates = np.array([np.nan] * len(x_train))

			# args.n_hidden = 100
			# for p2 in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
			# 	print('n_mix:',p2)
			# # 	for p1 in [1,2,3,4,5,6,7,8,9]:
			# 	args.n_mix = p2
			# # 		args.n_layers = p1 
			# 	estimates, est_slice = get_estimates(args, x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), slices)
			# 	# # test_est,  est_slice = estimate(args, x_test=x_test)
			# 	# # test_est  = np.median(test_est, 0)[:, est_slice[product]]
			# 	estimates = np.median(estimates, 0)#[:, est_slice[product]]
			# # estimates = np.array([np.nan] * len(x_train))
			# assert(0)

			estimates, est_slice = get_estimates(args, x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy(), slices)
			print(est_slice)
			print(slices)
			print(np.array(estimates).shape)
			# dist = (np.array(uppers_ens)-np.array(estimates)) / np.array(estimates)
			# orig = np.array(estimates).copy()
			estimates = np.median(estimates, 0)#[:, est_slice[product]]
			print(estimates.shape)
			# uppers = estimates + estimates*np.mean(dist,0)#np.mean(uppers_ens, 0)
			# lowers = estimates - estimates*np.mean(dist,0)#np.mean(lowers_ens, 0)
			# print('shape',uppers.shape)
			# e, u, l, y, a, b = map(np.array, zip(*sorted(zip(estimates, uppers, lowers, y_test, np.max(orig+dist*orig, 0), np.min(orig-dist*orig,0)), key=lambda k:k[0])))


			# print(((u > y) & (l < y)).sum() / len(y), ' (',((u > y) & (l < y)).sum(), '/', len(y), ')')
			# print(((a > y) & (b < y)).sum() / len(y), ' (',((a > y) & (b < y)).sum(), '/', len(y), ')')
			
			# plt.plot(y)
			# plt.plot(e)
			# l1 = plt.plot(u)
			# plt.plot(l, color=l1[0].get_color())
			# # l1 = plt.plot(a)
			# # plt.plot(b, color=l1[0].get_color())
			# plt.yscale('log')
			# plt.show()
			# m = lambda q1, q2: np.exp(np.abs(np.log(q1/q2)))-1
			# e = e.flatten()
			# y = y.flatten()
			# u = u.flatten()
			# a = a.flatten()
			# x,y = m(e,y), m(e,u)
			# def outlier(z):
			# 	mean = np.mean(z)
			# 	std  = np.std(z)
			# 	print(mean, std, z.min(), z.max())
			# 	return (z < (mean+2*std)) & (z > (mean-2*std))
			# import seaborn as sns
			# # plt.scatter(x,y)
			# mask = outlier(x)
			# mask = mask & outlier(y)
			# print(mask.sum())
			# sns.regplot(np.log10(x[mask]),np.log10(y[mask]))
			# # plt.xscale('log')
			# # plt.yscale('log')

			# # plt.xlim(x.min(), x.max())
			# # plt.ylim(y.min(), y.max())
			# plt.show()
			# assert(0)

			# j=13
			# print(mdsa(y_test[:, j], estimates[:, j]), waves[j])
			# assert(0)
			if 0:

				std  = np.std(y_data, 0)
				norm = np.mean(y_data, 0) / std 
				from .spectrum_rgb import get_spectrum_cmap 
				cmap = get_spectrum_cmap()
				bands = get_sensor_bands(args.sensor, args)

				import matplotlib.animation as animation
				fig = plt.figure()
				# plt.ion()
				# plt.show()
				# plt.pause(1e-9)
				extra_args = ["-tune", "zerolatency", "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white"]
				ani_writer = animation.writers['ffmpeg_file'](fps=5, extra_args=extra_args)
				ani_tmp  = Path('.').joinpath('tmp')
				ani_tmp.mkdir(parents=True, exist_ok=True)
				ani_writer.setup(fig, Path('.').joinpath(f'{args.product}_spectra.mp4').as_posix(), dpi=100, frame_prefix=ani_tmp.joinpath('_').as_posix(), clear_temp=False)

				from scipy.optimize import curve_fit
				def exponential(x, a, b, c):
					return a * np.exp(-b*x) + c 

				plt.scatter(
					[mdsa(y_test[i][None,:], estimates[i][None,:]) for i in range(y_test.shape[0])],
					[mdsa(y_test[i][None,:], y_orig[i][None,:]) for i in range(y_test.shape[0])],
				)
				plt.xscale('log')
				plt.yscale('log')
				plt.show()

				errs = [mdsa(y_test[i][None,:], estimates[i][None,:]) for i in range(y_test.shape[0])]
				errs = np.argsort(errs)[::-1]

				print(np.array(locs)[data_idxs[n_train+n_valid:]][:, errs[:10]])
				for i in errs:#range(y_test.shape[0]):
					# x_range = np.linspace(np.min(bands), np.max(bands), 50)
					
					plt.clf()
					plt.plot(bands, y_test[i], label='Measured')
					plt.plot(bands, estimates[i], label='Estimated')
					plt.plot(bands, y_orig[i], label='Original')
					print(mdsa(y_test[i][None,:], estimates[i][None,:]))
					try:
						y_estim, _ = curve_fit(exponential, np.arange(len(bands)), y_test[i])
						new_exp = exponential(np.arange(len(bands)), *y_estim)
						err1 = rmse(y_test[i][None,:], new_exp[None,:])
						err2 = mdsa(y_test[i][None,:], new_exp[None,:])
						print([y_estim, err1, err2])
						plt.plot(bands, new_exp, label='Fitted')
					except Exception as e: print(e)
					plt.legend()

					plt.twinx()
					plt.plot(bands, x_test[i], color='k')
					plt.show()

					# plt.ylim((1e-6, 10))
					# plt.yscale('log')

					# plt.pause(1e-9)
					# ani_writer.grab_frame()
					# ani_writer._run()
				assert(0)

				items = [('Error', [mdsa(y_test[:,i], estimates[:,i]) for i in range(y_test.shape[1])])]
				# items = [('Mean', np.mean(y_data,0)), ('Std', std), ('Std/Mean', norm)]
				colors = [cmap.to_rgba(nm) for nm in bands]
				fig, axes = plt.subplots(1, len(items), sharey=False, figsize=(5*len(items),5*1))
				fig.add_subplot(111, frameon=False)
				plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
				plt.title(r'$a_{ph}\ Variability$', fontsize=18)
				plt.title(r'$%s$' % args.product.replace('ph','_{ph}').replace('d','_{nap}').replace('g','_{cdom}'), fontsize=24)
				for i, (name, val) in enumerate(items):
					ax = axes[i] if type(axes) is list else axes
					ax.bar(bands, val, 5, color=colors)

					ax.set_ylabel(r'$\mathbf{%s}$' % name, fontsize=20)
					ax.set_xlabel(r'$\mathbf{Wavelength\ [nm]}$', fontsize=20)
					ax.tick_params(labelsize=18)
				# plt.subplots_adjust(wspace=0.25)
				plt.tight_layout()
				kwargs = {
					'bbox_inches' : 'tight',
					'pad_inches'  : 0.1,
				}
				plt.savefig(f'{args.sensor}_{args.product}_error.png', dpi=100, **kwargs)
				plt.show()
				assert(0)

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
			bands  = get_sensor_bands(args.sensor, args)
			n_wvl  = len(bands)
			s, args.sat_bands = args.sat_bands, False
			bench  = run_benchmarks(args, args.sensor, x_test_bench, y_test_bench, {p:slices[p] for p in product}, silent=False, x_train=x_train_bench, y_train=y_train_bench, gridsearch=False, with_ml=False)
			# bench  = run_benchmarks(args, args.sensor, x_test, y_test, {p:slices[p] for p in product}, silent=False, x_train=x_train, y_train=y_train, gridsearch=False, with_ml=False)
			args.sat_bands = s
			bdict  = bench
			labels = get_labels(bands, slices, y_test.shape[1])

			edge_colors = None
			if False and args.product in ['aph', 'ad', 'ag']:
				from .spectrum_rgb import get_spectrum_cmap, rgb_to_hsl, hsl_to_rgb
				cmap = get_spectrum_cmap()
				color = [cmap.to_rgba(nm) for nm in bands]
				
				# def pastelize(r,g,b,a=None): 
				# 	h,s,l = rgb_to_hsl(r,g,b)
				# 	print(h,s,l)
				# 	return hsl_to_rgb(h,s,l) + ((a,) if a is not None else ())

				# color = [pastelize(*c) for c in color]
				lighten = lambda x, pct=0.35: x + (1-x) * pct 
				# darken  = lambda x, pct=0.35: (x-pct) / (1-pct) 
				darken  = lambda x, pct=0.25: x - x*pct
				
				edge_colors = [(darken(r), darken(g), darken(b), a) for r,g,b,a in color]
				color   = [(lighten(r), lighten(g), lighten(b), a) for r,g,b,a in color]

				# color  = [(r,g,b,0.3) for r,g,b,a in color]
				# color  = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:lightish green', 'xkcd:reddish', 'xkcd:bluish purple']
			else:
				color  = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay']#, 'xkcd:bluish purple', 'xkcd:reddish']

			y_log  = np.hstack([np.log10(y_test[:, slices[p]]) for p in product]).T 

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

					target     = [443, 590, 620, 670] if product[0] == 'a*ph' else [443, 530,] if product[0] == 'aph' else [443,482,561,655]
					waves      = np.array(get_sensor_bands(args.sensor, args))
					valid_wvls = [waves[find_wavelength(w, waves)] for w in target]
					valid_plot = [w in valid_wvls for w in waves]
					n_target   = len(target)

					if len(waves) == len(color):
						color = [c for c, valid in zip(color, valid_plot) if valid]
						edge_colors = [e for e, valid in zip(edge_colors, valid_plot) if valid]

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


				labels = [(p,label) for label in labels for p in product if p in label]
				print('Plotting labels:', [l for i,l in enumerate(labels) if valid_plot[i]])

				if not len(bench):
					bench = [[None] * y_log.shape[1]]
				
				# fig, axes = plt.subplots(2, 2, sharey=False, figsize=(8,7))
				bench_order = None
				if n_plots > 3 or 'chl' in product:
					bench_order = ['Smith_Blend', 'OC6', 'Mishra_NDCI', 'Gons_2band', 'Gilerson_2band']
					# bench_order = ['OC3', 'XGB', 'SVM', 'MLP', 'KNN']
					bench_order = [b for b in bench_order if b in bdict]
					n_plots = len(bench_order) + 1
					fig, axes = plt.subplots(2, (n_plots+1)//2, sharey=False, figsize=(5*((n_plots+1)//2),10))
					# fig, axes = plt.subplots(n_target, n_plots, sharey=False, figsize=(n_plots*5,n_target*5+2))

				else:
					bench_order = ['QAA', 'GIOP']
					# bench_order = [b for b in bench_order if b in bdict]
					n_plots = len(bench_order) + 1
					# n_target, n_plots = n_plots, n_target
					fig, axes = plt.subplots(n_target, n_plots, sharey=False, figsize=(5*n_plots,5*n_target))
					# fig, axes = plt.subplots(1, 1, sharey=False, figsize=(5,5))
					# axes = [axes] * n_target
					# n_target, n_plots = n_plots, n_target

				print('Order:', bench_order)
				print('n plots:', n_plots)
				# if (n_target > 1 and n_plots > 1) or n_plots > 3: 
				axes = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
				print('n axes:', len(axes))
				print('n target:',n_target)
				fig.add_subplot(111, frameon=False)
				plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
				# axes[0][1].axis('off')
				curr_idx = 0
				# benches = [{k:v[:,i] for k,v in bench.items()} for i in range(slices[product].start, slices[product].stop)]
				# y_test = np.array([y_test[:, slices[p]] for p in product])

				# estimates = np.array([estimates[:, slices[p]] for p in product])

				color_idx = 0

				statbox_y_orig = []
				statbox_y_est  = []
				for plt_idx, (title, y_true, y_orig, y_model) in enumerate(zip(labels, y_log, y_test.T, estimates.T)):
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

						color_idx = i
						curr_idx += 1
						
						l_kws = {'color': color[color_idx], 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
						s_kws = {'alpha': 0.4, 'color': color[color_idx]}

						if edge_colors is not None:
							s_kws.update({'edgecolor': edge_colors[color_idx], 'lw': 1})

						if i:
							if bench_order is not None and len(bench_order):
								order = bench_order[i-1]
								print(i, 'Fetching', order, title)
								label = [b for b in bench if order in b]
								if len(label) > 1:
									label = [b for b in label if title in b]
									assert(len(label) == 1), [label, order, title] 
								if len(label) == 0: continue
								label   = label[0]
								y_bench = bench[label]
								# y_bench, label = [b for b in benches if order in b[1]][0]
								label = label.replace('Mishra_','').replace('Gons_2band', 'Gons').replace('Gilerson_2band', 'GI2B')

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
							if bench_order is not None and len(bench_order):
								[i.set_linewidth(5) for i in ax.spines.values()]
							y_model = y_model.flatten()
							y_est = np.log10(y_model)
							y_est_orig = y_model.copy()
							label = 'MDN_{A}'
							label = 'MDN-I'
			
						method_lbls.append(label.split(' ')[0])
						x_lbl = product.replace('chl', 'Chl\\textit{a}').replace('aph', '\\textit{a}_{ph}')
						y_lbl = product.replace('chl', 'Chl\\textit{a}^{e}').replace('aph', '\\textit{\^a}_{ph}')

						# if (hasattr(args, 'fix_tchl') and not args.fix_tchl) or (hasattr(args, 'keep_tchl') and args.keep_tchl):
						# 	prod_lbl = prod_lbl.replace('Chla', 'TChla')
						unit = r'[mg\ m^{-3}]' if product=='chl' else r'[g\ m^{-3}]' if product == 'tss' else r'[m^{-1}]'

						if curr_idx == 1: 
							plt.ylabel(fr'$\mathbf{{{y_lbl}\ '+unit + r'}$', fontsize=20, fontweight='bold', labelpad=10)
							# plt.xlabel(r'$\mathbf{%s}$' % (r'In\ situ\ '+fr'{prod_lbl}\ '+unit) +'\n'+ r'$\small{\mathit{N\small{=}}%s}$'%len(y_true), fontsize=20, fontweight='bold', labelpad=10)
							plt.xlabel(r'$\mathbf{%s}$' % (r''+fr'{x_lbl}\ '+unit), fontsize=20, fontweight='bold', labelpad=10)
							plot_title = get_sensor_label(args.sensor).replace('-',' ').replace(' ', '\ ')
							# plt.title(r'$\mathbf{\underline{\large{In\ Situ}}}$', fontsize=24, y=1.05)
							# plot_title = 'In\ Situ'
							# plot_title = 'Type\ A'
							# plt.title(r'$\mathbf{\underline{\large{%s}}}$'%plot_title + '\n' + r'$\small{\mathit{N\small{=}}%s}$'%len(y_true), fontsize=24, y=1.06)
							# plt.title(r'$\small{\mathit{N\small{=}}%s}$'%len(y_true), fontsize=24, y=1.06)

						if product not in ['chl', 'tss']:
							wvl = int(get_sensor_bands(args.sensor, args)[plt_idx])
							# prod_lbl += f'(\small{{{wvl}nm}})'

							if i == (n_plots-1):
								ax2 = ax.twinx()
								ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
								ax2.grid(False)
								ax2.set_yticklabels([])
								ax2.set_ylabel(fr'$\mathbf{{{wvl}nm}}$', fontsize=20)

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

						sns.regplot(y_true[valid], y_est[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True)#, label=f'{wvl}nm')#,label=r'$\mathrm{%s}$'%label.split(' ')[0])
						# sns.plotpoint(y_true[valid], y_est[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=False, label=f'{wvl}nm')#,label=r'$\mathrm{%s}$'%label.split(' ')[0])
						# col = color[((color_idx+1)%n_plots) if n_plots > 1 else (color_idx+1)]
						col = color[color_idx]
						kde = sns.kdeplot(y_true[valid], y_est[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=col)
						kde.collections[2].set_alpha(0)

						if len(valid.flatten()) != valid.sum():
							ax.scatter(y_true[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (~valid).sum())
							ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

						# if curr_idx == 1:
						add_identity(ax, ls='--', color='k', zorder=20)
						add_stats_box(ax, y_orig, y_est_orig)

						statbox_y_orig.append(y_orig)
						statbox_y_est.append(y_est_orig)
						# if curr_idx <= n_plots:
						ax.set_title(r'$\mathbf{\large{%s}}$' % label.split(' ')[0].replace('_2band','\ 2\ Band'), fontsize=18)

						ax.tick_params(labelsize=18)
						ax.grid('on', alpha=0.3)
					color_idx += 1

				# labels = [f'{int(get_sensor_bands(args.sensor, args)[i])}nm' for i in np.where(valid_plot)[0]]
				# add_stats_box(ax, np.array(statbox_y_orig).T, np.array(statbox_y_est).T, label=labels)
				# ax.legend(loc='lower right', labels=labels)

				plt.tight_layout()
				# plt.subplots_adjust(wspace=0.35)
				plt.savefig(f'scatters/{args.product}_{args.sensor}_scatter_{n_train}train.png', dpi=200, bbox_inches='tight', pad_inches=0.1,)
				plt.show()


	# Otherwise, train a model with all data (if not already existing)
	else:
		x_data, y_data, slices, locs = get_data(args)

		if args.model_lbl == 'exclude20140909':
			set_name, set_idxs = locs.T
			set_idxs = set_idxs.astype(int) 

			i = (set_name == 'GLERL') & np.isin(set_idxs, [23, 24, 25, 26, 27])
			i|= (set_name == 'Caren') & np.isin(set_idxs, [77,78,79,80,81,82,83,84,85])

			x_data = x_data[~i]
			y_data = y_data[~i]
			print(f'Removed {i.sum()} Lake Erie samples')

		get_estimates(args, x_data, y_data, output_slices=slices)

