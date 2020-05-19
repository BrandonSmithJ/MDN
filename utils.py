from .transformers import CustomUnpickler, RatioTransformer, ColumnTransformer
from .meta import get_sensor_bands, SENSOR_BANDS
from .metrics import bias, slope, rmsle, mape, mae, rmse, r_squared, sspb, mdsa 
from .parameters import update, hypers
from .__version__ import __version__

from collections import defaultdict as dd
from datetime import datetime as dt
from pathlib import Path
from tqdm import trange
import pickle as pkl
import numpy as np 
import hashlib


def find_wavelength(k, waves):
	''' Index of closest wavelength '''
	return np.abs(np.array(waves) - k).argmin() 	


def closest_wavelength(k, waves, validate=True): 
	''' Value of closest wavelength '''
	w = waves[find_wavelength(k, waves)]	
	assert(not validate or (abs(k-w) <= 5)), f'Needed {k}nm, but closest was {w}nm in {waves}'
	return w 


def get_tile_Rrs(filename, sensor, allow_neg=True):
	''' Gather the correct Rrs bands from a given tile '''
	from netCDF4 import Dataset
	with Dataset(filename, 'r') as tile:
		if 'geophysical_data' in tile.groups.keys():
			tile = tile['geophysical_data']
	
		tile_key = 'Rrs_' if any(['Rrs_' in k for k in tile.variables]) else 'Rw'
		tile_wvl = [int(v.replace(tile_key, '')) for v in tile.variables if tile_key in v]
		bands    = [closest_wavelength(b, tile_wvl) for b in get_sensor_bands(sensor)]
		div      = np.pi if tile_key == 'Rw' else 1
		
		Rrs = [tile[f'{tile_key}{b}'][:] / div for b in bands]
		if not allow_neg:
			for v in Rrs:
				v[v <= 0] = np.nan 
		return bands, Rrs
		

def store_scaler(scaler, args=[], kwargs={}):
	return (scaler, args, kwargs)


def add_noise(X, Y, percent=0.10):
	X += X * percent * np.random.normal(size=X.shape) + X * percent * np.random.choice([-1,1,0], size=(X.shape[0], 1))#(len(x_batch),1)) / 10 
	# Y += Y * percent * np.random.normal(size=Y.shape) + Y * percent * np.random.choice([-1,1,0], size=(Y.shape[0], 1))#(len(y_batch),1)) / 10
	return X, Y 


def add_identity(ax, *line_args, **line_kwargs):
	''' 
	Add 1 to 1 diagonal line to a plot.
	https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
	
	Usage: add_identity(plt.gca(), color='k', ls='--')
	'''
	identity, = ax.plot([], [], *line_args, **line_kwargs)
	
	def callback(axes):
		low_x, high_x = ax.get_xlim()
		low_y, high_y = ax.get_ylim()
		lo = max(low_x,  low_y)
		hi = min(high_x, high_y)
		identity.set_data([lo, hi], [lo, hi])

	callback(ax)
	ax.callbacks.connect('xlim_changed', callback)
	ax.callbacks.connect('ylim_changed', callback)

	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points', 
		'xycoords'   : 'axes fraction', 
		'fontname'   : 'monospace', 
		'xytext'     : (0,0), 
		'zorder'     : 25, 	
		'va'         : 'top', 
		'ha'         : 'left', 
	}
	ax.annotate(r'$\mathbf{1:1}$', xy=(0.87,0.99), size=11, **ann_kwargs)


def add_stats_box(ax, y_true, y_est, metrics=[mdsa, sspb, slope], bottom_right=False, x=0.025, y=0.97, fontsize=16):
	''' Add statistics box to a plot '''
	import matplotlib.pyplot as plt
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	longest = max([len(metric.__name__) for metric in metrics])
	statbox = []
	percent = ['mape', 'sspb', 'mdsa']
	for metric in metrics:
		name  = metric.__qualname__
		label = metric.__name__.replace('SSPB', 'Bias').replace('MSA', 'Error')
		diff  = longest-len(label)
		space = r''.join([r'\ ']*diff + [r'\thinspace']*diff)
		prec  = 1 if name in percent else 3
		stat  = f'{metric(y_true, y_est):.{prec}f}'
		perc  = r'$\small{\mathsf{\%}}$' if name in percent else ''

		statbox.append(rf'$\mathtt{{{label}}}{space}:$ {stat}{perc}')

	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points', 
		'xycoords'   : 'axes fraction', 
		'fontname'   : 'monospace', 
		'xytext'     : (0,0), 
		'zorder'     : 25, 	
		'va'         : 'top', 
		'ha'         : 'left', 
		'bbox'       : {
			'facecolor' : 'white',
			'edgecolor' : 'black', 
			'alpha'     : 0.7,
		}
	}

	ann = ax.annotate('\n'.join(statbox), xy=(x,y), size=fontsize, **ann_kwargs)
	# ann.set_bbox(dict(facecolor='white', alpha=.7, edgecolor='black', zorder=26))

	# Switch location to bottom right corner
	if bottom_right:
		plt.gcf().canvas.draw()
		bbox_orig = ann.get_tightbbox(plt.gcf().canvas.renderer).transformed(ax.transAxes.inverted())

		new_x = 1 - (bbox_orig.x1 - bbox_orig.x0) + x
		new_y = bbox_orig.y1 - bbox_orig.y0 + (1 - y)
		ann.set_x(new_x)
		ann.set_y(new_y)
		ann.xy = (new_x - 0.04, new_y + 0.06)
	return ann 
	

def line_messages(messages):
	''' 
	Allow multiline message updates via tqdm. 
	Need to call print() after the tqdm loop, 
	equal to the number of messages which were
	printed via this function (to reset cursor).
	
	Usage:
		for i in trange(5):
			messages = [i, i/2, i*2]
			line_messages(messages)
		for _ in range(len(messages)): print()
	'''
	for i, m in enumerate(messages, 1):
		trange(1, desc=str(m), position=i, bar_format='{desc}')


def get_labels(wavelengths, slices, n_out=None):
	''' 
	Helper to get label for each target output. Assumes 
	that any variable in <slices> which has more than a 
	single slice index, will have an associated wavelength
	label. 

	Usage:
		wavelengths = [443, 483, 561, 655]
		slices = {'bbp':slice(0,4), 'chl':slice(4,5), 'tss':slice(5,6)}
		n_out  = 5
		labels = get_labels(wavelengths, slices, n_out) 
			# labels -> ['bbp443', 'bbp483', 'bbp561', 'bbp655', 'chl']
	'''
	return [k + ('%i'%wavelengths[i] if (v.stop - v.start) > 1 else '') 
			for k,v in sorted(slices.items(), key=lambda s: s[1].start)
			for i   in range(v.stop - v.start)][:n_out]	


def generate_config(args, create=True):
	''' Create a config file for the current settings, and
		store in a folder corresponding to its hash '''
	dependents = [getattr(act, 'dest', '') for group in [hypers, update] for act in group._group_actions]
	dependents+= ['x_scalers', 'y_scalers']

	config = [f'Version: {__version__}']
	config+= [''.join(['-']*len(config[-1]))]
	others = [''.join(['-']*len(config[-1]))]

	for k,v in sorted(args.__dict__.items(), key=lambda z: z[0]):
		if k in ['x_scalers', 'y_scalers']:
			v = [(s[0].__name__,)+s[1:] for s in v]

		if k in dependents: config.append(f'{k}: {v}')
		else:               others.append(f'{k}: {v}')
				
	config = '\n'.join(config)
	others = '\n'.join(others)
	uid    = hashlib.sha256(config.encode('utf-8')).hexdigest()
	folder = Path(__file__).parent.resolve().joinpath(args.model_loc, args.sensor, args.model_lbl, uid)
	conf_file = folder.joinpath('config')

	if create:
		folder.mkdir(parents=True, exist_ok=True)
		
		if not conf_file.exists():
			with conf_file.open('w+') as f:
				f.write(f'Created: {dt.now()}\n{config}\n{others}')
	elif not conf_file.exists():
		print(config)
	return folder 


def load_data(keys, locs, _wavelengths):
	''' 
	Load data from [<locs>] using <keys> as the columns. 
	Only loads data which has all the bands defined by 
	<wavelengths> (if necessary, e.g. for Rrs or bbp).
	First key is assumed to be the x_data, remaining keys
	(if any) are y_data.

	Usage:
		# Here, data/loc/Rrs.csv, data/loc/Rrs_wvl.csv, data/loc/bbp.csv, 
		# and data/chl.csv all exist, with the correct wavelengths available
		# for Rrs and bbp (which is determined by Rrs_wvl.csv)
		keys = ['Rrs', 'bbp', '../chl']
		locs = 'data/loc'
		wavelengths = [443, 483, 561, 655]
		load_data(keys, locs, wavelengths) # -> [Rrs443, Rrs483, Rrs561, Rrs665], 
												[bbp443, bbp483, bbp561, bbp655, chl], 
												{'bbp':slice(0,4), 'chl':slice(4,5)}
	'''
	def loadtxt(name, loc, wavelengths): 
		''' Wrapper for np.loadtxt, ensuring data is shaped [samples, features] '''
		dloc = Path(loc).joinpath(f'{name}.csv')

		try:
			assert(dloc.exists()), (f'Key {name} does not exist at {loc} ({dloc})') 
			data = np.loadtxt(dloc, delimiter=',', dtype=float if name not in ['../Dataset', '../meta'] else str, comments=None)
			if len(data.shape) == 1:
				data = data[:, None]

			if data.shape[1] > 1 and data.dtype.type is not np.str_:
				valid = get_valid(name, loc, wavelengths)
				data  = data[:, valid]

				# If we want to get all data, regardless of if bands are available...
				# new_data = [[np.nan]*len(data)] * len(wavelengths)
				# wvls  = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
				# idxs  = np.abs(wvls - np.atleast_2d(wavelengths)).argmin(0)
				# valid = np.abs(wvls - np.atleast_2d(wavelengths)).min(0) < 2

				# for j, (i, v) in enumerate(zip(idxs, valid)):
				# 	print(j, i, v)
				# 	if v: new_data[j] = data[:, i]
				# data = np.array(new_data).T

			return data 
		except Exception as e:
			assert(0)
			if dloc.exists():
				print(f'Error fetching {name} from {loc}: {e}')
			if name not in ['Rrs']:# ['../chl', '../tss', '../cdom']:
				return np.array([]).reshape((0,0))
			assert(0), e

	def get_valid(name, loc, wavelengths, margin=2):
		''' Dataset at <loc> must have all bands in <wavelengths> within 10nm '''
		if 'HYPER' in str(loc): margin=1
		wvls = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
		assert(np.all([np.abs(wvls-w).min() <= margin for w in wavelengths])), (
			'%s is missing wavelengths. \n%s needed,\n%s found' % (loc, wavelengths, wvls.flatten()))
		
		if len(wvls) != len(wavelengths):
			valid = np.abs(wvls - np.atleast_2d(wavelengths)).min(1) < margin
			assert(valid.sum() == len(wavelengths)), [wvls[valid].flatten(), wavelengths]
			return valid 
		return np.array([True] * len(wavelengths))

	x_data = []
	y_data = []
	l_data = []
	for loc in np.atleast_1d(locs):
		try:
			loc_data = [loadtxt(key, loc, _wavelengths) for key in keys]
			print(f'N={len(loc_data[0]):>5} | {loc.parts[-1]} / {loc.parts[-2]} ({[np.isfinite(ld).all(1).sum() for ld in loc_data[1:]]})')
			assert(all([len(l) in [len(loc_data[0]), 0] for l in loc_data])), dict(zip(keys, map(np.shape, loc_data)))

			if all([l.shape[1] == 0 for l in loc_data[1:]]):
				print(f'Skipping dataset {loc}: missing all features')
				continue

			x_data  += [loc_data.pop(0)]
			y_data  += [loc_data]
			l_data  += [loc] * len(x_data[-1])

		except Exception as e:
			assert(0), e
			# Allow invalid datasets if there are multiple to be fetched
			print(f'Error {loc}: {e}')
			if len(np.atleast_1d(locs)) == 1:
				raise e
	assert(len(x_data) > 0 or len(locs) == 0), 'No datasets are valid with the given wavelengths'
	assert(all([x.shape[1] == x_data[0].shape[1] for x in x_data])), f'Differing number of {keys[0]} wavelengths: {[x.shape for x in x_data]}'

	# Determine the number of features each key should have
	slices = []
	for i, key in enumerate(keys[1:]):
		shapes = [y[i].shape[1] for y in y_data]
		slices.append(max(shapes))

		for x, y in zip(x_data, y_data):
			if y[i].shape[1] == 0:
				y[i] = np.full((x.shape[0], max(shapes)), np.nan)
		assert(all([y[i].shape[1] == y_data[0][i].shape[1] for y in y_data])), f'{key} shape mismatch: {[y.shape for y in y_data]}'

	# Drop any missing features
	drop = []
	for i, s in enumerate(slices):
		if s == 0:
			print(f'Dropping {keys[i+1]}: feature has no samples available')
			drop.append(i)

	print()
	slices = np.cumsum([0] + [s for i,s in enumerate(slices) if i not in drop])
	keys   = [k for i,k in enumerate(keys[1:]) if i not in drop]
	for y in y_data:
		y = [z for i,z in enumerate(y) if i not in drop]

	# Combine everything together
	l_data = np.vstack(l_data)
	x_data = np.vstack(x_data)
	y_data = np.vstack([np.hstack(y) for y in y_data])
	assert(slices[-1] == y_data.shape[1]), [slices, y_data.shape]
	assert(y_data.shape[0] == x_data.shape[0]), [x_data.shape, y_data.shape]

	slices = {k.replace('../','') : slice(slices[i], s) for i,(k,s) in enumerate(zip(keys, slices[1:]))}
	return x_data, y_data, slices, l_data



def get_valid(x_train, y_train, x_test, y_test, slices, within_train=False, partial_nan=False, other=None):
	''' 
	Filter the given training and testing data to only include samples
	are valid. By default, valid samples include all which are not nan,
	and greater than zero (for all target features). 
	- partial_nan=True can be set to allow a sample as valid if _any_ 
	  target features are not nan and greater than zero (only for testing data).
	- within_train=True can be set to also remove any testing samples 
	  which are not within the training data set bounds, with respect to 
	  all features.
	- "other" is an optional set of parameters which will be pruned with the 
	  test sets (i.e. passing a list of indices will return the indices which
	  were kept)
	Multiple data sets can also be passed simultaneously as a list to the 
	respective parameters, in order to filter the same samples out of all
	data sets (e.g. OLI and S2B data, containing same samples but different
	bands, can be filtered so they end up with the same samples relative to
	each other).
	'''
	
	# Allow multiple sets to be given, and align them all to the same sample subset
	if type(x_train) is not list: x_train = [x_train]
	if type(y_train) is not list: y_train = [y_train]
	if type(x_test)  is not list: x_test  = [x_test]
	if type(y_test)  is not list: y_test  = [y_test]
	if other is not None and type(other) is not list: other = [other]

	train_both = [x_train, y_train]
	test_both  = [x_test,  y_test]
	total_data = train_both + test_both 
	set_length = [len(fullset) for fullset in total_data]
	set_shape  = [[len(subset) for subset in fullset] for fullset in total_data]
	assert(np.all([length == len(x_train) for length in set_length])), set_length
	assert(np.all([[shape == len(fullset[0]) for shape in shapes] 
					for shapes, fullset in zip(set_shape, total_data)])), set_shape		
	assert(other is None or all([len(o) == len(x_test[0]) for o in other])), \
					[len(o) for o in other]

	# Hydrolight simulated CDOM is incorrectly scaled
	# if 'cdom' in slices:
	# 	s = slices['cdom']
	# 	for y in y_train:
	# 		y[:, s] = y[:, s] * 0.18

	# Ensure only positive / finite training features
	valid = np.ones(len(x_train[0])).astype(np.bool)
	for i, fullset in enumerate(train_both):
		for subset in fullset:
			subset[np.isnan(subset)] = -999.
			subset[subset <= 0]      = np.nan 
			has_nan = np.any if partial_nan and i else np.all 
			valid = np.logical_and(valid, has_nan(np.isfinite(subset), 1))

	x_train = [x[valid] for x in x_train]
	y_train = [y[valid] for y in y_train]
	print(f'Removed {len(valid)-valid.sum()} invalid training samples samples ({valid.sum()} remaining)')

	# Ensure only positive / finite testing features, but
	# allow the possibility of some nan values in y_test,
	# if the sample has other non-nan values
	valid = np.ones(len(x_test[0])).astype(np.bool)
	for i, fullset in enumerate(test_both):
		for subset in fullset:
			subset[np.isnan(subset)] = -999.
			subset[subset <= 0]      = np.nan 
			has_nan = np.any if partial_nan and i else np.all 
			valid   = np.logical_and(valid, has_nan(np.isfinite(subset), 1))

	# Use only test samples which are within training bounds
	if within_train:
		for train_full, test_full in [[x_train, x_test], [y_train, y_test]]:
			for train_sub, test_sub in zip(train_full, test_full):
				for feature, f_min, f_max in zip(test_sub.T, train_sub.min(0), train_sub.max(0)):
					valid = np.logical_and(valid, np.logical_and(
							np.logical_or(np.isnan(feature), feature >= f_min),
							np.logical_or(np.isnan(feature), feature <= y_max)))

	assert(valid.sum()), 'All test points have values outside of training data range'
	x_test = [x[valid] for x in x_test]
	y_test = [y[valid] for y in y_test]
	#np.savetxt('bool.csv', valid, delimiter=',')
	#assert(0)

	print(f'Removed {len(valid)-valid.sum()} invalid testing samples ({valid.sum()} remaining)')

	if len(x_train) == 1:
		ret = (x_train[0], y_train[0], x_test[0], y_test[0])
	else:
		ret = (x_train, y_train, x_test, y_test)

	if other is not None:
		other = [np.array(o)[valid] for o in other]
		ret += (other,)
	return ret



def store_pkl(filename, output):
	''' Helper to write pickle file '''
	with Path(filename).open('wb') as f:
		pkl.dump(output, f)
	return output

def read_pkl(filename):
	''' Helper to read pickle file '''
	with Path(filename).open('rb') as f:
		return CustomUnpickler(f).load()

def cache(filename, recache=False):
	''' Decorator for caching function outputs '''
	path = Path(filename)

	def wrapper(function):
		def inner(*args, **kwargs):
			if not recache and path.exists():
				return read_pkl(path)
			return store_pkl(path, function(*args, **kwargs))
		return inner
	return wrapper



def get_data(args):
	sensor  = args.sensor.split('-')[0]
	product = args.product.split(',')

	get_dataset = lambda path, p: Path(path.as_posix().replace(f'/{sensor}','').replace(f'/{p}.csv','')).stem

	if product[0] == 'all':
		product = ['chl', 'tss','cdom']#, 'ad', 'ag', 'aph']# + ['aph', 'a*ph', 'apg', 'a'] 

	test_path   = Path(args.test_loc)
	test_folder = []
	test_keys   = ['Rrs']
	for p in product:
		if p in ['chl', 'tss', 'cdom']:
			p = f'../{p}'

		folders = [get_dataset(path, p) for path in test_path.glob(f'*/{sensor}/{p.replace("*", "[*]")}.csv')]

		if p == 'aph':
			folders = [f for f in folders if f not in ['Gurlin3', 'PACE']]
		
		if p == '../chl':
			folders = [f for f in folders if f not in ['Bunkei_a', 'Caren']]
			if args.test_set == 'paper': # MSI / OLCI paper
				folders = ['Sundar', 'UNUSED/Taihu_old', 'UNUSED/Taihu2', 'UNUSED/Schalles_old', 'SeaBASS2', 'Vietnam'] 

		if p == 'bb_p':
			folders = ['UNUSED/248']
			assert(args.use_sim), 'Must use simulated data to estimate bbp'

		test_folder += [f for f in folders if f not in test_folder]
		if p not in test_keys: test_keys.append(p)

	assert(len(test_folder)), f'No datasets found for {product}'
	assert(len(test_keys)), f'No variables found for {product}'

	if args.use_sim:
		assert(args.product == 'bb_p'), 'Can only use simulated for bbp at the moment'
		train_keys = ['Rrs', 'bb_p', 'a_p', '../chl', '../tss', '../cdom']
	else:
		train_keys = test_keys

	assert([k1 == k2 for k1, k2 in zip(test_keys, train_keys)]), (
		f'Train/test key mismatch: {train_keys}, {test_keys}')
	
	train_data_loc = args.train_loc 
	test_data_loc  = args.test_loc 

	# Use test data as training data
	if not args.use_sim:
		train_keys     = test_keys 
		train_folder   = test_folder
		train_data_loc = test_data_loc
		print('Using in situ samples as training data.')
	else: 
		train_folder = ['848']
		test_folder = []

	# Use a cached data set if available, to speed up fetching
	cache_name = 'Cache/%s_%s_%s_%s.pkl' % (args.sensor, 
		'%sratio' % ('' if (hasattr(args, 'no_ratio') and not args.no_ratio) or (hasattr(args, 'add_ratio') and args.add_ratio) else 'no'),
		'all_ins' if type(train_folder) is list else train_folder, 
		'all_ins' if type(test_folder)  is list else test_folder)
	cache_name = Path(__file__).parent.resolve().joinpath(cache_name)
	cache_name.parent.mkdir(parents=True, exist_ok=True)

	@cache(cache_name, recache=(hasattr(args, 'no_cache') and args.no_cache) or (hasattr(args, 'use_cache') and not args.use_cache))
	def fetch(args, align):
		join_loc  = lambda loc, folder, sensor: Path(loc).joinpath(folder, sensor.replace('ETM-pan','ETM800').split('-')[0])
		train_loc = [join_loc(train_data_loc, f, args.sensor) for f in np.atleast_1d(train_folder)]
		test_loc  = [join_loc(test_data_loc,  f, args.sensor) for f in np.atleast_1d(test_folder )]

		if align is not None and (type(align) is str or len(align)):
			align = np.atleast_1d(align)
			test_locs_align = [[join_loc(args.test_loc, f, a) for f in np.atleast_1d(test_folder )] for a in align]

		if args.use_sim:
			x_train, y_train, train_slices, train_locs = load_data(train_keys, train_loc, get_sensor_bands(args.sensor, args))
			if len(test_loc):
				x_test,  y_test,  test_slices,  test_locs  = load_data(test_keys,  test_loc,  get_sensor_bands(args.sensor, args)) 
			else:
				x_test = x_train
				y_test = y_train
				test_slices = train_slices
				test_locs = train_locs
		else:
			x_train, y_train, train_slices, train_locs = \
			x_test,  y_test,  test_slices,  test_locs  = load_data(train_keys, train_loc, get_sensor_bands(args.sensor, args))
		print(len(y_test),'initial size')
		
		y_train_bak = y_train.copy()
		idx_count = dd(int)
		test_idxs = []
		test_locs = np.array([l[0].as_posix().split('/')[-2] for l in test_locs])
		for l in test_locs:
			test_idxs.append(idx_count[l])
			idx_count[l] += 1

		if align is not None and len(align):
			x_tests = []
			y_tests = []
			for a, align_loc in zip(align, test_locs_align):
				x_test_a, y_test_a, _, _ = load_data(test_keys, align_loc, get_sensor_bands(a, args))
				x_tests.append(x_test_a)
				y_tests.append(y_test_a) 
			x_trains, y_trains, x_tests, y_tests, locs = get_valid([x_train]+x_tests, [y_train]+y_tests, [x_test]+x_tests, [y_test]+y_tests, train_slices, other=[test_locs, test_idxs])
			x_train = x_trains[0]
			y_train = y_trains[0]
			x_test  = x_tests[0]
			y_test  = y_tests[0]

		else:
			x_train, y_train, x_test, y_test, locs = get_valid(x_train, y_train, x_test, y_test, train_slices, partial_nan=len(test_keys) > 2, other=[test_locs, test_idxs])
			# locs = (test_locs, test_idxs)
		idxs = locs[1]
		locs = locs[0]

		print('\nFinal counts:')
		print('\n'.join([f'N={num:>5} | {loc}' for loc, num in zip(*np.unique(locs, return_counts=True))]))
		print(f'\tTotal: {len(locs)}')

		# Correct chl data for pheopigments
		if 'chl' in args.product and ((hasattr(args, 'fix_tchl') and args.fix_tchl) or (hasattr(args, 'keep_tchl') and not args.keep_tchl)):

			fix = np.ones(len(x_train)).astype(np.bool)
			old = y_train.copy()

			full_idxs = np.where(locs == 'Sundar')[0]
			dataset   = np.loadtxt(join_loc(args.test_loc, 'Sundar', 'Dataset.csv'), delimiter=',', dtype=str)[idxs[full_idxs]]
			fix[full_idxs[dataset == 'ACIX_Krista']] = False
			fix[full_idxs[dataset == 'ACIX_Moritz']] = False

			import pandas as pd 
			full_idxs = np.where(locs == 'SeaBASS2')[0]
			meta = pd.read_csv(join_loc(args.test_loc, 'SeaBASS2', 'meta.csv')).iloc[idxs[full_idxs]]
			lonlats = meta[['east_longitude', 'west_longitude', 'north_latitude', 'south_latitude']].apply(lambda v: v.apply(lambda v2: v2.split('||')[0]))
			assert(lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all().all())
			lonlats = lonlats.apply(lambda v: pd.to_numeric(v.apply(lambda v2: v2.split('::')[1].replace('[deg]','')), 'coerce'))
			lonlats = lonlats[['east_longitude', 'north_latitude']].to_numpy()
			i = np.logical_and(lonlats[:,0] < -117, lonlats[:,1] > 32)
			fix[full_idxs[i]] = False

			fix[y_train[:,0] > 80] = False

			print(f'Correcting {fix.sum()} / {len(fix)} samples')
			coef = [0.04, 0.776, 0.015, -0.00046, 0.000004]
			# coef = [-0.12, 0.9, 0.001]
			x    = y_train[fix,0][:,None]
			y_train[fix,0] = np.sum(np.array(coef) * x ** np.arange(len(coef)), 1, keepdims=False)
			# print(y_train.min(), (y_train < 0).sum())
			# print(locs[y_train.flatten() < 0])

			# import matplotlib.pyplot as plt
			# plt.scatter(old, y_train)
			# plt.xlabel('Old')
			# plt.ylabel('New')
			# plt.xscale('log')
			# plt.yscale('log')
			# add_identity(plt.gca(), color='k', ls='--')
			# plt.xlim((y_train[y_train > 0].min()/10, y_train.max()*10))
			# plt.ylim((y_train[y_train > 0].min()/10, y_train.max()*10))
			# plt.show()
			# assert(0)

		return x_train, y_train, x_test, y_test, train_slices, (idxs, locs)

	aligned = getattr(args, 'align', None)
	if aligned is not None:
		aligned = args.align.split(',')
		if 'all' in aligned: 
			aligned = [s for s in SENSOR_LABELS.keys() if s != 'HYPER']
	x_train, y_train, x_test, y_test, train_slices, locs = fetch(args, aligned)

	# if args.verbose:
	# 	print('\nBenchmarks:')
	# 	print_benchmarks(args.sensor, x_test[:,:len(get_sensor_bands(args.sensor, args))], y_test, train_slices)
	# 	print()

	if not args.use_sim:
		x_test = x_train
		y_test = y_train
	else:
		assert(not np.any(np.isnan(x_train)))
		assert(not np.any(np.isnan(y_train)))

	# n_targets = y_test.shape[1]
	# print('Min/Max Test Y: ', list(zip(np.nanmin(y_test, 0).round(2), np.nanmax(y_test, 0).round(2))))
	# print('Min/Max Train Y:', list(zip(y_train.min(0).round(2)[:n_targets], y_train.max(0).round(2)[:n_targets])))
	# print('Shapes:',x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	# print()
	return x_train, y_train, x_test, y_test, train_slices, locs


def bagging_subset(args, x_train, y_train, x_scalers, y_scalers, percent=0.75):
	# Return a subset of the training data to allow bagging
	# using_ratio = (hasattr(args, 'no_ratio') and not args.no_ratio) or (hasattr(args, 'add_ratio') and args.add_ratio) #or len(x_train.T) > 12

	pct  = 0.75
	rows = np.arange(len(x_train))
	cols = np.arange(len(x_train.T))

	# if using_ratio:
	# 	cols = np.arange(len(x_train_orig.T)-len(get_sensor_bands(args.sensor, args))) + len(get_sensor_bands(args.sensor, args))

	nrow = max(int(len(rows)*pct), 100)
	ncol = min(int(len(cols)*0.75), len(cols)-1)

	np.random.shuffle(rows)
	np.random.shuffle(cols)
	# rows = np.random.choice(rows, size=nrow, replace=True)

	x_remain = x_train[rows[nrow:]]
	y_remain = y_train[rows[nrow:]]
	x_train  = x_train[rows[:nrow]]
	y_train  = y_train[rows[:nrow]]

	# if using_ratio:
	# 	x_scalers = [store_scaler(ColumnTransformer, args=[np.append(np.arange(len(get_sensor_bands(args.sensor, args))), cols[:ncol])])] + x_scalers
	return x_scalers, y_scalers, x_train, y_train, x_remain, y_remain
