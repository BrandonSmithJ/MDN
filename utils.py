from .meta import get_sensor_bands, ANCILLARY, PERIODIC
from .parameters import update, hypers, flags, get_args
from .__version__ import __version__

from collections import defaultdict as dd
from importlib import import_module 
from datetime import datetime as dt
from pathlib import Path
from tqdm import trange

import pickle as pkl
import numpy as np 
import hashlib, re, warnings, functools, sys, zipfile


def ignore_warnings(func):
	''' Decorator to silence all warnings (Runtime, User, Deprecation, etc.) '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore') 
			return func(*args, **kwargs)
	return helper 


def find_wavelength(k, waves, validate=True, tol=5):
	''' Index of closest wavelength '''
	waves = np.array(waves)
	w = np.atleast_1d(k)
	i = np.abs(waves - w[:, None]).argmin(1) 
	assert(not validate or (np.abs(w-waves[i]).max() <= tol)), f'Needed {k}, but closest was {waves[i]} in {waves} ({np.abs(w-waves[i]).max()} > {tol})'
	return i.reshape(np.array(k).shape)


def closest_wavelength(k, waves, validate=True, tol=5): 
	''' Value of closest wavelength '''
	waves = np.array(waves)
	return waves[find_wavelength(k, waves, validate, tol)]	


def safe_int(v):
	''' Parse int if possible, and return None otherwise '''
	try: return int(v)
	except: return None


def get_wvl(nc_data, key):
	''' Get all wavelengths associated with the given key, available within the netcdf '''
	wvl = [safe_int(v.replace(key, '')) for v in nc_data.variables.keys() if key in v]
	return np.array(sorted([w for w in wvl if w is not None]))


def line_messages(messages, nbars=1):
	''' 
	Allow multiline message updates via tqdm. 
	Need to call print() after the tqdm loop, 
	equal to the number of messages which were
	printed via this function (to reset cursor).
	
	nbars is the number of tqdm bars the line
	messages come after.

	Usage:
		nbars = 2
		for i in trange(5):
			for j in trange(5, leave=False):
				messages = [i, i/2, i*2]
				line_messages(messages, nbars)
		for _ in range(len(messages) + nbars - 1): print()
	'''
	for _ in range(nbars): print()
	for m in messages: print('\033[K' + str(m))
	sys.stdout.write('\x1b[A'.join([''] * (nbars + len(messages) + 1)))


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
	return [k + (f'{wavelengths[i]:.0f}' if (v.stop - v.start) > 1 else '') 
			for k,v in sorted(slices.items(), key=lambda s: s[1].start)
			for i   in range(v.stop - v.start)][:n_out]	


def compress(path, overwrite=False):
	''' Compress a folder into a .zip archive '''
	if overwrite or not path.with_suffix('.zip').exists():
		with zipfile.ZipFile(path.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
			for item in path.rglob('*'):
				zf.write(item, item.relative_to(path))


def uncompress(path, overwrite=False):
	''' Uncompress a .zip archive '''
	if overwrite or not path.exists():
		if path.with_suffix('.zip').exists():
			with zipfile.ZipFile(path.with_suffix('.zip'), 'r') as zf:
				zf.extractall(path)


class CustomUnpickler(pkl.Unpickler):
	''' Ensure the classes are found, without requiring an import '''
	_transformers = [p.stem for p in Path(__file__).parent.joinpath('transformers').glob('*Transformer.py')]
	_warned       = False

	def find_class(self, module, name):
		# pathlib/pickle doesn't correctly deal with instantiating
		# a system-specific path on the opposite system (e.g. WindowsPath
		# on a linux OS). Instead, we just provide the general Path class. 
		if name in ['WindowsPath', 'PosixPath']:
			return Path 

		elif name in self._transformers:
			module   = Path(__file__).parent.stem
			imported = import_module(f'{module}.transformers.{name}')
			return getattr(imported, name)
			
		elif name == 'TransformerPipeline':
			from .transformers import TransformerPipeline
			return TransformerPipeline

		return super().find_class(module, name)

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


def using_feature(args, flag):
	''' 
	Certain hyperparameter flags have a yet undecided default value,
	which means there are two possible names: using the feature, or 
	not using it. This method simply combines both into a single 
	boolean signal, which indicates whether to add the feature. 
	For example:
	 	use_flag = hasattr(args, 'use_ratio') and args.use_ratio
		no_flag  = hasattr(args, 'no_ratio') and not args.no_ratio 
		signal   = use_flag or no_flag  # if true, we add ratios
	becomes
		signal = using_feature(args, 'ratio') # if true, we add ratios  
	'''
	flag = flag.replace('use_', '').replace('no_', '')
	assert(hasattr(args,f'use_{flag}') or hasattr(args, f'no_{flag}')), f'"{flag}" flag not found'
	return getattr(args, f'use_{flag}', False) or not getattr(args, f'no_{flag}', True)


def split_data(x_data, other_data=[], n_train=0.5, n_valid=0, seed=None, shuffle=True):
	''' 
	Split the given data into training, validation, and testing 
	subsets, randomly shuffling the original data order.
	'''
	if not isinstance(other_data, list): other_data = [other_data]

	data    = [d.iloc if hasattr(d, 'iloc') else d for d in [x_data] + other_data]
	random  = np.random.RandomState(seed)
	idxs    = np.arange(len(x_data))
	if shuffle: random.shuffle(idxs)

	# Allow both a percent to be passed in, as well as an absolute number
	if 0 < n_train <= 1: n_train = int(n_train * len(idxs)) 
	if 0 < n_valid <= 1: n_valid = int(n_valid * len(idxs))
	assert((n_train+n_valid) <= len(x_data)), \
		'Too many training/validation samples requested: {n_train}, {n_valid} ({len(x_data)} available)'

	train = [d[ idxs[:n_train] ]                for d in data]
	valid = [d[ idxs[n_train:n_valid+n_train] ] for d in data]
	test  = [d[ idxs[n_train+n_valid:] ]        for d in data]

	# Return just the split x_data if no other data was given
	if len(data) == 1:
		train = train[0]
		valid = valid[0]
		test  = test[0]

	# If no validation data was requested, just return train/test
	if n_valid == 0:
		return train, test 
	return train, valid, test


@ignore_warnings
def mask_land(data, bands, threshold=0.2, verbose=False):
	''' Modified Normalized Difference Water Index, or NDVI if 1500nm+ is not available '''
	green = closest_wavelength(560,  bands, validate=False)
	red   = closest_wavelength(700,  bands, validate=False)
	nir   = closest_wavelength(900,  bands, validate=False)
	swir  = closest_wavelength(1600, bands, validate=False)
	
	b1, b2 = (green, swir) if swir > 1500 else (red, nir) if red != nir else (min(bands), max(bands))
	i1, i2 = find_wavelength(b1, bands), find_wavelength(b2, bands)
	n_diff = lambda a, b: np.ma.masked_invalid((a-b) / (a+b))
	if verbose: print(f'Using bands {b1} & {b2} for land masking')
	# import matplotlib.pyplot as plt 
	# plt.clf()
	# plt.imshow(n_diff(data[..., i1], data[..., i2]).filled(fill_value=threshold-1))
	# plt.colorbar()
	# plt.show()
	# assert(0)
	return n_diff(data[..., i1], data[..., i2]).filled(fill_value=threshold-1) <= threshold


@ignore_warnings
def _get_tile_wavelengths(nc_data, key, sensor, allow_neg=True, landmask=False, args=None):
	''' Return the Rrs/rhos data within the netcdf file, for wavelengths of the given sensor '''
	has_key = lambda k: any([k in v for v in nc_data.variables])
	wvl_key = f'{key}_' if has_key(f'{key}_') or key != 'Rrs' else 'Rw' # Polymer stores Rw=Rrs*pi

	if has_key(wvl_key):
		avail = get_wvl(nc_data, wvl_key)
		bands = [closest_wavelength(b, avail) for b in get_sensor_bands(sensor, args)]
		div   = np.pi if wvl_key == 'Rw' else 1
		data  = np.ma.stack([nc_data[f'{wvl_key}{b}'][:] / div for b in bands], axis=-1)
		
		if not allow_neg: data[data <= 0] = np.nan
		if landmask:      data[ mask_land(data, bands) ] = np.nan

		return bands, data.filled(fill_value=np.nan)
	return [], np.array([])

def get_tile_data(filenames, sensor, allow_neg=True, rhos=False, anc=False, **kwargs):
	''' Gather the correct Rrs/rhos bands from a given scene, as well as ancillary features if necessary '''
	from netCDF4 import Dataset

	filenames = np.atleast_1d(filenames) 
	features  = ['rhos' if rhos else 'Rrs'] + (ANCILLARY if anc or rhos else [])
	data      = {}
	available = []

	# Some sensors use different bands for their rhos models 
	if rhos and '-rho' not in sensor: sensor += '-rho'

	args = get_args(sensor=sensor, **kwargs)
	for filename in filenames:
		with Dataset(filename, 'r') as nc_data:
			if 'geophysical_data' in nc_data.groups.keys():
				nc_data = nc_data['geophysical_data']
	
			for feature in features:
				if feature not in data:
					if feature in ['Rrs', 'rhos']:
						bands, band_data = _get_tile_wavelengths(nc_data, feature, sensor, allow_neg, landmask=rhos, args=args)
	
						if len(bands) > 0: 
							assert(len(band_data.shape) == 3), \
								f'Different shape than expected: {band_data.shape}'
							data[feature] = band_data
	
					elif feature in nc_data.variables:
						var = nc_data[feature][:]
						assert(len(var.shape) == 2), f'Different shape than expected: {var.shape}'
	
						if feature in PERIODIC:
							assert(var.min() >= -180 and var.max() <= 180), \
								f'Need to adjust transformation for variables not within [-180,180]: {feature}=[{var.min()}, {var.max()}]'
							data[feature] = np.stack([
								np.sin(2*np.pi*(var+180)/360),
								np.cos(2*np.pi*(var+180)/360),
							], axis=-1)
						else: data[feature] = var
	
	# Time difference should just be 0: we want estimates for the exact time of overpass
	if 'time_diff' in features:
		assert(features[0] in data), f'Missing {features[0]} data: {list(data.keys())}'
		data['time_diff'] = np.zeros_like(data[features[0]][:, :, 0])

	assert(len(data) == len(features)), f'Missing features: Found {list(data.keys())}, Expecting {features}'
	return bands, np.dstack([data[f] for f in features])


def generate_config(args, create=True, verbose=True):
	''' 
	Create a config file for the current settings, and store in
	a folder location determined by certain parameters: 
		MDN/model_loc/sensor/model_lbl/model_uid/config
	"model_uid" is computed within this function, but a value can 
	also be passed in manually via args.model_uid in order to allow
	previous MDN versions to run.
	'''
	root = Path(__file__).parent.resolve().joinpath(args.model_loc, args.sensor, args.model_lbl)

	# Can override the model uid in order to allow prior MDN versions to be run
	if hasattr(args, 'model_uid'):
		if args.verbose: print(f'Using manually set model uid: {args.model_uid}')
		return root.joinpath(args.model_uid)

	# Hash is always dependent upon these values
	dependents = [getattr(act, 'dest', '') for group in [hypers, update] for act in group._group_actions]
	dependents+= ['x_scalers', 'y_scalers']

	# Hash is only partially dependent upon these values, assuming operation changes when using a feature
	#  - 'use_' flags being set cause dependency
	#  - 'no_'  flags being set remove dependency
	# This allows additional flags to be added without breaking prior model compatibility
	partials = [getattr(act, 'dest', '') for group in [flags] for act in group._group_actions]

	config = [f'Version: {__version__}', '', 'Dependencies']
	config+= [''.join(['-']*len(config[-1]))]
	others = ['', 'Configuration']
	others+= [''.join(['-']*len(others[-1]))]

	for k,v in sorted(args.__dict__.items(), key=lambda z: z[0]):
		if k in ['x_scalers', 'y_scalers']: 
			cinfo = lambda s, sarg, skw: getattr(s, 'config_info', lambda *a, **k: '')(*sarg, **skw)
			cfmt  = lambda *cargs: f' # {cinfo(*cargs)}' if cinfo(*cargs) else '' 
			v = '\n\t' + '\n\t'.join([f'{(s[0].__name__,) + s[1:]}{cfmt(*s)}' for s in v]) # stringify scaler and its arguments
		
		if k in partials and using_feature(args, k): 
			                     config.append(f'{k:<18}: {v}')
		elif k in dependents:    config.append(f'{k:<18}: {v}')
		else:                    others.append(f'{k:<18}: {v}') 

	config = '\n'.join(config) # Model is dependent on some arguments, so they change the uid
	others = '\n'.join(others) # Other arguments are stored for replicability
	ver_re = r'(Version\: \d+\.\d+)(?:\.\d+\n)' # Match major/minor version within subgroup, patch/dashes within pattern
	h_str  = re.sub(ver_re, r'\1.0\n', config)  # Substitute patch version for ".0" to allow patches within the same uid
	uid    = hashlib.sha256(h_str.encode('utf-8')).hexdigest()
	folder = root.joinpath(uid)
	c_file = folder.joinpath('config')
	uncompress(folder) # Unzip the archive if necessary
	
	if args.verbose: 
		print(f'Using model path {folder}')

	if create:
		folder.mkdir(parents=True, exist_ok=True)
		
		if not c_file.exists():
			with c_file.open('w+') as f:
				f.write(f'Created: {dt.now()}\n{config}\n{others}')
	elif not c_file.exists() and verbose:
		print('\nCould not find config file with the following parameters:')
		print('\t'+config.replace('\n','\n\t'),'\n')
	return folder 


def _load_datasets(keys, locs, wavelengths, allow_missing=False):
	''' 
	Load data from [<locs>] using <keys> as the columns. 
	Only loads data which has all the bands defined by 
	<wavelengths> (if necessary, e.g. for Rrs or bbp).
	First key is assumed to be the x_data, remaining keys
	(if any) are y_data.
	  - allow_missing=True will allow datasets which are missing bands
	    to be included in the returned data

	Usage:
		# Here, data/loc/Rrs.csv, data/loc/Rrs_wvl.csv, data/loc/bbp.csv, 
		# and data/chl.csv all exist, with the correct wavelengths available
		# for Rrs and bbp (which is determined by Rrs_wvl.csv)
		keys = ['Rrs', 'bbp', '../chl']
		locs = 'data/loc'
		wavelengths = [443, 483, 561, 655]
		_load_datasets(keys, locs, wavelengths) # -> [Rrs443, Rrs483, Rrs561, Rrs665], 
												 [bbp443, bbp483, bbp561, bbp655, chl], 
											 	 {'bbp':slice(0,4), 'chl':slice(4,5)}
	'''
	def loadtxt(name, loc, required_wvl): 
		''' Error handling wrapper over np.loadtxt, with the addition of wavelength selection'''
		dloc = Path(loc).joinpath(f'{name}.csv')
		
		# TSS / TSM / SPM are synonymous
		if 'tss' in name and not dloc.exists():
			dloc = Path(loc).joinpath(f'{name.replace("tss","tsm")}.csv')

			if not dloc.exists():
				dloc = Path(loc).joinpath(f'{name.replace("tsm","spm")}.csv')

		# CDOM is just an alias for a_cdom(443) or a_g(443)
		if 'cdom' in name and not dloc.exists():
			dloc = Path(loc).joinpath('ag.csv')
			required_wvl = [443]

		try:
			required_wvl = np.array(required_wvl).flatten()
			assert(dloc.exists()), (f'Key {name} does not exist at {loc} ({dloc})') 

			data = np.loadtxt(dloc, delimiter=',', dtype=float if name not in ['../Dataset', '../meta', '../datetime'] else str, comments=None)
			if len(data.shape) == 1: data = data[:, None]

			if data.shape[1] > 1 and data.dtype.type is not np.str_:

				# If we want to get all data, regardless of if bands are available...
				if allow_missing:
					new_data = [[np.nan]*len(data)] * len(required_wvl)
					wvls  = np.loadtxt(Path(loc).joinpath(f'{dloc.stem}_wvl.csv'), delimiter=',')[:,None]
					idxs  = np.abs(wvls - np.atleast_2d(required_wvl)).argmin(0)
					valid = np.abs(wvls - np.atleast_2d(required_wvl)).min(0) < 2

					for j, (i, v) in enumerate(zip(idxs, valid)):
						if v: new_data[j] = data[:, i]
					data = np.array(new_data).T
				else:
					data = data[:, get_valid(dloc.stem, loc, required_wvl)]

			if 'cdom' in name and dloc.stem == 'ag':
				data = data[:, find_wavelength(443, required_wvl)].flatten()[:, None]
			return data 
		except Exception as e:
			if name not in ['Rrs']:# ['../chl', '../tss', '../cdom']:
				if dloc.exists():
					print(f'\n\tError fetching {name} from {loc}:\n{e}')
				return np.array([]).reshape((0,0))
			raise e

	def get_valid(name, loc, required_wvl, margin=2):
		''' Dataset at <loc> must have all bands in <required_wvl> within <margin>nm '''
		if 'HYPER' in str(loc): margin=1

		# First, validate all required wavelengths are within the margin of an available wavelength
		wvls  = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
		check = np.array([np.abs(wvls-w).min() <= margin for w in required_wvl])
		assert(check.all()), '\n\t\t'.join([
			f'{name} is missing {(~check).sum()} wavelengths:',
			f'Needed  {required_wvl}', f'Found   {wvls.flatten()}', 
			f'Missing {required_wvl[~check]}', ''])

		# First, validate available wavelengths are within the margin of the required wavelengths
		valid = np.array([True] * len(required_wvl))
		if len(wvls) != len(required_wvl):
			valid = np.abs(wvls - np.atleast_2d(required_wvl)).min(1) <= margin
			assert(valid.sum() == len(required_wvl)), [wvls[valid].flatten(), required_wvl]

		# Then, ensure the order of the available wavelengths are the same as the required
		if not all([w1 == w2 for w1,w2 in zip(wvls[valid], required_wvl)]):
			valid = [np.abs(wvls.flatten() - w).argmin() for w in required_wvl]
			assert(len(np.unique(valid)) == len(valid) == len(required_wvl)), [valid, wvls[valid].flatten(), required_wvl]
		return valid 

	locs = [Path(loc).resolve() for loc in np.atleast_1d(locs)]
	print('\n-------------------------')
	print(f'Loading data for sensor {locs[0].parts[-1]}, and targets {[v.replace("../","") for v in keys[1:]]}')
	if allow_missing:
		print('Allowing data regardless of whether all bands exist')

	x_data = []
	y_data = []
	l_data = []
	for loc in locs:
		try:
			loc_data = [loadtxt(key, loc, wavelengths) for key in keys]
			print(f'\tN={len(loc_data[0]):>5} | {loc.parts[-1]} / {loc.parts[-2]} ({[np.isfinite(ld).all(1).sum() if ld.dtype.type is not np.str_ else len(ld) for ld in loc_data[1:]]})')
			assert(all([len(l) in [len(loc_data[0]), 0] for l in loc_data])), dict(zip(keys, map(np.shape, loc_data)))

			if all([l.shape[1] == 0 for l in loc_data[(1 if len(loc_data) > 1 else 0):]]):
				print(f'Skipping dataset {loc}: missing all features')
				continue

			x_data  += [loc_data.pop(0)]
			y_data  += [loc_data]
			l_data  += list(zip([loc.parent.name] * len(x_data[-1]), np.arange(len(x_data[-1]))))

		except Exception as e:
			# assert(0), e
			# Allow invalid datasets if there are multiple to be fetched
			print(f'\nError fetching {loc}:\n\t{e}')
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

	slices = np.cumsum([0] + [s for i,s in enumerate(slices) if i not in drop])
	keys   = [k for i,k in enumerate(keys[1:]) if i not in drop]
	for y in y_data:
		y = [z for i,z in enumerate(y) if i not in drop]

	# Combine everything together
	l_data = np.vstack(l_data)
	x_data = np.vstack(x_data)

	if len(keys) > 0:
		y_data = np.vstack([np.hstack(y) for y in y_data])
		assert(slices[-1] == y_data.shape[1]), [slices, y_data.shape]
		assert(y_data.shape[0] == x_data.shape[0]), [x_data.shape, y_data.shape]
	slices = {k.replace('../','') : slice(slices[i], s) for i,(k,s) in enumerate(zip(keys, slices[1:]))}
	print(f'\tTotal prior to filtering: {len(x_data)}')

	# Fit exponential function to ad and ag values, and eliminate samples with too much error
	for product in ['ad', 'ag']:
		if product in slices:
			from .metrics import mdsa
			from scipy.optimize import curve_fit

			exponential = lambda x, a, b, c: a * np.exp(-b*x) + c 
			remove      = np.zeros_like(y_data[:,0]).astype(bool)

			for i, sample in enumerate(y_data):
				sample = sample[slices[product]]
				assert(len(sample) > 5), f'Number of bands should be larger, when fitting exponential: {product}, {sample.shape}'
				assert(len(sample) == len(wavelengths)), f'Sample size / wavelengths mismatch: {len(sample)} vs {len(wavelengths)}'
				
				if np.all(np.isfinite(sample)) and np.min(sample) > -0.1:
					try:
						x = np.array(wavelengths) - np.min(wavelengths)
						params, _  = curve_fit(exponential, x, sample, bounds=((1e-3, 1e-3, 0), (1e2, 1e0, 1e1)))
						new_sample = exponential(x, *params)

						# Should be < 10% error between original and fitted exponential 
						if mdsa(sample[None,:], new_sample[None,:]) < 10:
							y_data[i, slices[product]] = new_sample
						else: remove[i] = True # Exponential could be fit, but error was too high
					except:   remove[i] = True # Sample deviated so much from a smooth exponential decay that it could not be fit
				# else:         remove[i] = True # NaNs / negatives in the sample

			# Don't actually drop them yet, in case we are fetching all samples regardless of nan composition
			x_data[remove] = np.nan
			y_data[remove] = np.nan
			l_data[remove] = np.nan

			if remove.sum():
				print(f'Removed {remove.sum()} / {len(remove)} samples due to poor quality {product} spectra')
				assert((~remove).sum()), f'All data removed due to {product} spectra quality...'

	return x_data, y_data, slices, l_data


def _filter_invalid(x_data, y_data, slices, allow_nan_inp=False, allow_nan_out=False, other=[]):
	''' 
	Filter the given data to only include samples which are valid. By 
	default, valid samples include all which are not nan, and greater 
	than zero (for all target features). 
	- allow_nan_inp=True can be set to allow a sample as valid if _any_ 
	  of a sample's input x features are not nan and greater than zero.
	- allow_nan_out=True can be set to allow a sample as valid if _any_ 
	  of a sample's target y features are not nan and greater than zero.
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
	if type(x_data) is not list: x_data = [x_data]
	if type(y_data) is not list: y_data = [y_data]
	if type(other)  is not list: other  = [other]

	both_data  = [x_data, y_data]
	set_length = [len(fullset) for fullset in both_data]
	set_shape  = [[len(subset) for subset in fullset] for fullset in both_data]

	assert(np.all([length == len(x_data) for length in set_length])), \
		f'Mismatching number of subsets: {set_length}'
	assert(np.all([[shape == len(fullset[0]) for shape in shapes] 
					for shapes, fullset in zip(set_shape, both_data)])), \
		f'Mismatching number of samples: {set_shape}'		
	assert(len(other) == 0 or all([len(o) == len(x_data[0]) for o in other])), \
		f'Mismatching number of samples within other data: {[len(o) for o in other]}'

	# Ensure only positive / finite testing features, but allow the
	# possibility of some nan values in x_data (if allow_nan_inp is
	# set) or in y_data (if allow_nan_out is set) - so long as the 
	# sample has other non-nan values in the respective feature set
	valid = np.ones(len(x_data[0])).astype(np.bool)
	for i, fullset in enumerate(both_data):
		for subset in fullset:
			subset[np.isnan(subset)] = -999.
			subset[np.logical_or(subset <= 0, not i and (subset >= 10))] = np.nan 
			has_nan = np.any if (i and allow_nan_out) or (not i and allow_nan_inp) else np.all 
			valid   = np.logical_and(valid, has_nan(np.isfinite(subset), 1))

	x_data = [x[valid] for x in x_data]
	y_data = [y[valid] for y in y_data]
	print(f'Removed {(~valid).sum()} invalid samples ({valid.sum()} remaining)')
	assert(valid.sum()), 'All samples have nan or negative values'

	if len(other) > 0:
		return x_data, y_data, [np.array(o)[valid] for o in other]
	return x_data, y_data


def get_data(args):
	''' Main function for gathering datasets '''
	np.random.seed(args.seed)
	sensor   = args.sensor.split('-')[0]
	products = args.product.split(',')
	bands    = get_sensor_bands(args.sensor, args)

	# Using Hydrolight simulated data
	if using_feature(args, 'sim'):
		assert(not using_feature(args, 'ratio')), 'Too much memory needed for simulated+ratios'
		data_folder = ['790']
		data_keys   = ['Rrs']+products #['Rrs', 'bb_p', 'a_p', '../chl', '../tss', '../cdom']
		data_path   = Path(args.sim_loc)

	else:
		if products[0] == 'all':
			products = ['chl', 'tss', 'cdom', 'ad', 'ag', 'aph']# + ['a*ph', 'apg', 'a'] 

		data_folder = []
		data_keys   = ['Rrs']
		data_path   = Path(args.data_loc)
		get_dataset = lambda path, p: Path(path.as_posix().replace(f'/{sensor}','').replace(f'/{p}.csv','')).stem

		for product in products:
			if product in ['chl', 'tss', 'cdom']:
				product = f'../{product}'
		
			# Find all datasets with the given product available
			safe_prod = product.replace('*', '[*]') # Prevent glob from getting confused by wildcard
			datasets  = [get_dataset(path, product) for path in data_path.glob(f'*/{sensor}/{safe_prod}.csv')]

			if product == 'aph':
				datasets = [d for d in datasets if d not in ['PACE']]
			
			if getattr(args, 'subset', ''):
				datasets = [d for d in datasets if d in args.subset.split(',')]
				
			data_folder += datasets
			data_keys   += [product]

	# Get only unique entries, while also preserving insertion order
	order_unique = lambda a: [a[i] for i in sorted(np.unique(a, return_index=True)[1])]
	data_folder  = order_unique(data_folder)
	data_keys    = order_unique(data_keys)
	assert(len(data_folder)), f'No datasets found for {products} within {data_path}/*/{sensor}'
	assert(len(data_keys)),  f'No variables found for {products} within {data_path}/*/{sensor}'
	
	sensor_loc = [data_path.joinpath(f, sensor) for f in data_folder]
	x_data, y_data, slices, sources = _load_datasets(data_keys, sensor_loc, bands, allow_missing=('-nan' in args.sensor) or (getattr(args, 'align', None) is not None))

	# Hydrolight simulated CDOM is incorrectly scaled
	if using_feature(args, 'sim') and 'cdom' in slices:
		y_data[:, slices['cdom']] *= 0.18

	# Allow data from one sensor to be aligned with other sensors (so the samples will be the same across sensors) 
	if getattr(args, 'align', None) is not None:
		assert('-nan' not in args.sensor), 'Cannot allow all samples via "-nan" while also aligning to other sensors'
		align = args.align.split(',')
		if 'all' in align: 
			align = [s for s in SENSOR_LABELS.keys() if s != 'HYPER']
		align_loc = [[data_path.joinpath(f, a.split('-')[0]) for f in data_folder] for a in align]

		print(f'\nLoading alignment data for {align}...')
		x_align, y_align, slices_align, sources_align = map(list,
			zip(*[_load_datasets(data_keys, loc, get_sensor_bands(a, args), allow_missing=True) for a, loc in zip(align, align_loc)]))
		
		x_data = [x_data] + x_align
		y_data = [y_data] + y_align

	# if -nan IS in the sensor label: do not filter samples; allow all, regardless of nan composition
	if '-nan' not in args.sensor:
		(x_data, *_), (y_data, *_), (sources, *_) = _filter_invalid(x_data, y_data, slices, other=[sources], allow_nan_out=not using_feature(args, 'sim') and len(data_keys) > 2)
			
	print('\nFinal counts:')
	print('\n'.join([f'\tN={num:>5} | {loc}' for loc, num in zip(*np.unique(sources[:, 0], return_counts=True))]))
	print(f'\tTotal: {len(sources)}')

	# Correct chl data for pheopigments
	if 'chl' in args.product and using_feature(args, 'tchlfix'):
		assert(not using_feature(args, 'sim')), 'Simulated data does not need TChl correction'
		y_data = _fix_tchl(y_data, sources, slices, data_path)

	return x_data, y_data, slices, sources


def _fix_tchl(y_data, sources, slices, data_path, debug=False):
	''' Very roughly correct chl for pheopigments '''
	import pandas as pd 

	dataset_name, sample_idx = sources.T 
	sample_idx.astype(int)

	fix = np.ones(len(y_data)).astype(np.bool)
	old = y_data.copy()

	set_idx = np.where(dataset_name == 'Sundar')[0]
	dataset = np.loadtxt(data_path.joinpath('Sundar', 'Dataset.csv'), delimiter=',', dtype=str)[sample_idx[set_idx]]
	fix[set_idx[dataset == 'ACIX_Krista']] = False
	fix[set_idx[dataset == 'ACIX_Moritz']] = False

	set_idx = np.where(data_lbl == 'SeaBASS2')[0]
	meta    = pd.read_csv(data_path.joinpath('SeaBASS2', 'meta.csv')).iloc[sample_idx[set_idx]]
	lonlats = meta[['east_longitude', 'west_longitude', 'north_latitude', 'south_latitude']].apply(lambda v: v.apply(lambda v2: v2.split('||')[0]))
	# assert(lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all().all()), lonlats[~lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all(1)]
	
	lonlats = lonlats.apply(lambda v: pd.to_numeric(v.apply(lambda v2: v2.split('::')[1].replace('[deg]','')), 'coerce'))
	lonlats = lonlats[['east_longitude', 'north_latitude']].to_numpy()

	# Only needs correction in certain areas, and for smaller chl magnitudes
	fix[set_idx[np.logical_and(lonlats[:,0] < -117, lonlats[:,1] > 32)]] = False
	fix[y_data[:,0] > 80] = False
	print(f'Correcting {fix.sum()} / {len(fix)} samples')

	coef = [0.04, 0.776, 0.015, -0.00046, 0.000004]
	# coef = [-0.12, 0.9, 0.001]
	y_data[fix, slices['chl']] = np.sum(np.array(coef) * y_data[fix, slices['chl']] ** np.arange(len(coef)), 1, keepdims=False)

	if debug:
		import matplotlib.pyplot as plt
		from .plot_utils import add_identity
		plt.scatter(old, y_data)
		plt.xlabel('Old')
		plt.ylabel('New')
		plt.xscale('log')
		plt.yscale('log')
		add_identity(plt.gca(), color='k', ls='--')
		plt.xlim((y_data[y_data > 0].min()/10, y_data.max()*10))
		plt.ylim((y_data[y_data > 0].min()/10, y_data.max()*10))
		plt.show()
	return y_data

