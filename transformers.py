from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.base import TransformerMixin
from sklearn import preprocessing
from itertools import combinations

from .meta import get_sensor_bands 
from .utils import using_feature

import pickle as pkl
import numpy as np 
import warnings


class CustomTransformer(TransformerMixin):
	''' Data transformer class which validates data shapes. 
		Child classes should override _fit, _transform, _inverse_transform '''
	_input_shape  = None 
	_output_shape = None

	def fit(self, X, *args, **kwargs):				 
		self._input_shape = X.shape[1]
		self._fit(X.copy(), *args, **kwargs)
		return self 

	def transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._input_shape)
		X = self._transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._output_shape)
		self._output_shape = X.shape[1]
		return X 

	def inverse_transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._output_shape)
		X = self._inverse_transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._input_shape)
		self._input_shape = X.shape[1]
		return X 

	@staticmethod
	def config_info(*args, **kwargs):                 return '' # Return any additional info to construct model config
	def _fit(self, X, *args, **kwargs):               pass
	def _transform(self, X, *args, **kwargs):         raise NotImplemented
	def _inverse_transform(self, X, *args, **kwargs): raise NotImplemented
	def _validate_shape(self, X, shape):              assert(shape is None or X.shape[1] == shape), \
		f'Number of data features changed: expected {shape}, found {X.shape[1]}'


class IdentityTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):         return X
	def _inverse_transform(self, X, *args, **kwargs): return X


class LogTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):         return np.log(X)
	def _inverse_transform(self, X, *args, **kwargs): return np.exp(X)


class NegLogTransformer(CustomTransformer):
	''' 
	Log-like transformation which allows negative values (Whittaker et al. 2005)
	http://fmwww.bc.edu/repec/bocode/t/transint.html
	'''
	def _transform(self, X, *args, **kwargs):         return np.sign(X) *  np.log(np.abs(X)  + 1)
	def _inverse_transform(self, X, *args, **kwargs): return np.sign(X) * (np.exp(np.abs(X)) - 1)


class ColumnTransformer(CustomTransformer):
	''' Reduce columns to specified selections (feature selection) '''
	def __init__(self, columns, *args, **kwargs):     self._c = columns 
	def _transform(self, X, *args, **kwargs):         return X[:, self._c]


class AUCTransformer(CustomTransformer):
	''' Area under the curve normalization '''
	def __init__(self, wavelengths, *args, **kwargs): self.wavelengths = wavelengths
	def _transform(self, X, *args, **kwargs):         return X/np.trapz(X, self.wavelengths, axis=1)[:, None]


class BaggingColumnTransformer(CustomTransformer):
	''' 
	Randomly select a percentage of columns to drop, always keeping the
	band features (first n_bands columns). Optionally, the last n_extra 
	columns will also be always kept as additional features.
	'''
	percent = 0.75

	def __init__(self, n_bands, *args, n_extra=0, **kwargs):
		self.n_bands = n_bands
		self.n_extra = n_extra

	def _fit(self, X, *args, **kwargs):
		shp  = X.shape[1] - self.n_bands
		ncol = int(shp*self.percent)
		cols = np.arange(shp-self.n_extra) + self.n_bands
		np.random.shuffle(cols)
		new_cols  = list(cols[:ncol])
		if self.n_extra: new_cols += list(X.shape[1]-(np.arange(self.n_extra)+1))
		self.cols = np.append(np.arange(self.n_bands), new_cols, 0)
		# print(f'Reducing bands from {shp} ({X.shape[1]} total) to {ncol} ({len(self.cols)} total) ({self.cols})')

	def _transform(self, X, *args, **kwargs):
		return X[:, self.cols.astype(int)]


class ExclusionTransformer(CustomTransformer):
	''' 
	Exclude certain columns from being transformed by the given transformer.
	The passed in transformer should be a transformer class, and exclude_slice can
	be any object which, when used to slice a numpy array, will give the 
	appropriate columns which should be excluded. So, for example:
		- slice(1)
		- slice(-3, None)
		- slice(1,None,2)
		- np.array([True, False, False, True])
		etc.
	'''
	def __init__(self, exclude_slice, transformer, transformer_args=[], transformer_kwargs={}):
		self.excl = exclude_slice
		self.transformer = transformer(*transformer_args, **transformer_kwargs)

	def _fit(self, X, *args, **kwargs):
		cols = np.arange(X.shape[1])
		cols = [c for c in cols if c not in cols[self.excl]]
		self.transformer.fit(X[:, cols])
		self.keep = cols
		return self

	def _transform(self, X, *args, **kwargs):
		Z = np.zeros_like(X)
		Z[:, self.keep] = self.transformer.transform(X[:, self.keep])
		Z[:, self.excl] = X[:, self.excl]
		return Z 

	def _inverse_transform(self, X, *args, **kwargs):
		Z = np.zeros_like(X)
		Z[:, self.keep] = self.transformer.inverse_transform(X[:, self.keep])
		Z[:, self.excl] = X[:, self.excl]
		return Z 


class RatioTransformer(CustomTransformer):	
	''' Add ratio features '''

	def __init__(self, wavelengths, *args, excl_Rrs=False, all_ratio=False, **kwargs):
		assert(not all_ratio or len(wavelengths) < 16), 'Too many features would be created with > 15 bands (i.e. > 1000 features)'
		self.wavelengths = list(wavelengths)
		self.all_ratio   = all_ratio
		self.excl_Rrs    = excl_Rrs
		self.labels      = []

	@staticmethod
	def config_info(*args, **kwargs):
		transformer = RatioTransformer(*args, **kwargs)
		return f'{transformer.get_n_features()} features: {transformer.labels}'

	def get_n_features(self):
		random_data = np.ones((2, len(self.wavelengths)))
		return self._transform(random_data).shape[-1]

	def _fit(self, X, *args, **kwargs):
		self.shape = X.shape[1]

	def _transform(self, X, *args, **kwargs):		 
		''' Ratios based on literature '''
		from .Benchmarks.utils import get_required, has_band, closest_wavelength
		self.labels = []

		x     = np.atleast_2d(X)
		x_new = []
		Rrs   = get_required(x, self.wavelengths, tol=11)

		def BR2(L1, L2):
			self.labels.append(f'{L1}/{L2}')
			return [Rrs(L1) / Rrs(L2)]

		def BR3(L1, L2, L3):
			self.labels.append(f'{L2}/{L1}-{L2}/{L3}')
			return [Rrs(L2)/Rrs(L1) - Rrs(L2)/Rrs(L3)]

		def LH(L1, L2, L3):
			self.labels.append(f'{L1}-{L2}-{L3}')
			c = (L3 - L2) / (L3 - L1)
			return [Rrs(L2) - c*Rrs(L1) - (1-c)*Rrs(L3)]

		def PCA(n_components=1):
			from sklearn.decomposition import PCA as PCA_sklearn
			values = PCA_sklearn(n_components).fit_transform( Rrs(None) )
			self.labels += [f'PCA{i}' for i in range(values.shape[-1])]
			return [values[:, i] for i in range(values.shape[-1])]

		features = (
			[(BR2, wvl) for wvl in combinations(self.wavelengths, 2)] + 
			[(BR3, wvl) for wvl in combinations(self.wavelengths, 3)] + 
			[(LH,  wvl) for wvl in combinations(self.wavelengths, 3)]  
		) if self.all_ratio else [
			(BR2, (745, 783)), 
			(BR2, (665, 705)),
			(BR2, (560, 705)),
			(BR2, (560, 665)),
			(BR2, (490, 560)),
			(BR2, (443, 490)),
			(BR2, (665, 620)),
			(BR2, (620, 560)),

			(LH, (490, 560, 665)),
			(LH, (665, 705, 745)),
			(LH, (705, 745, 783)),
			(LH, (560, 620, 665)),
		]

		for function, wvls in features:
			if all(has_band(wvl, self.wavelengths, tol=11) for wvl in wvls):
				x_new += function(*closest_wavelength(wvls, self.wavelengths, tol=11))

		# x_new.append((R705 - R665) / (R705 + R665)); self.labels.append('NDCI') # NDCI
		# x_new.append((R560 - R1613) / (R560 + R1613)); self.labels.append('MNDWI') # NDWI
		# x_new.append((1/R665 - 1/R705) / (1/R783 - 1/R705)); self.labels.append('YangBI') # Yang band index

		for xv in x_new: assert(np.all(np.isfinite(xv))), X[~np.isfinite(xv)]
		assert(len(x_new) == len(self.labels)), f'Mismatch between features and labels: {len(x_new)} vs {len(self.labels)}'
		if self.excl_Rrs:
			return np.hstack([v.flatten()[:, None] for v in x_new])
		return np.append(x, np.hstack([v.flatten()[:, None] for v in x_new]), -1)

	def _inverse_transform(self, X, *args, **kwargs): 
		return np.array(X)[:, :self.shape]


class KBestTransformer(CustomTransformer):
	''' Select the top K features, based on mutual information with the target variable. 
		When multiple target variables exist, each target variable receives an even share
		of the features (i.e. 3 targets, K=6 -> K=2 for each target) '''
	def __init__(self, n_features, *args, **kwargs):
		self.n_features = n_features

	def _fit(self, X, y, *args, **kwargs):
		self.k_transformer = []
		for i in range(y.shape[-1]):
			valid = np.isfinite(y[..., i])
			self.k_transformer.append( SelectKBest(mutual_info_regression, k=self.n_features // y.shape[-1]) )
			self.k_transformer[-1].fit(X[valid], y[valid, i])
		return self 

	def _transform(self, X, *args, **kwargs):
		return X[..., sorted(np.unique([np.argsort(t.scores_)[-t.k:] for t in self.k_transformer]))]


class TanhTransformer(CustomTransformer):
	''' tanh-estimator (Hampel et al. 1986; Latha & Thangasamy, 2011) '''
	scale = 0.01

	def _fit(self, X, *args, **kwargs):
		m = np.median(X, 0)
		d = np.abs(X - m)

		a = np.percentile(d, 70, 0)
		b = np.percentile(d, 85, 0)
		c = np.percentile(d, 95, 0)

		Xab = np.abs(X)
		Xsi = np.sign(X)
		phi = np.zeros(X.shape)
		idx = np.logical_and(0 <= Xab, Xab < a)
		phi[idx] = X[idx]
		idx = np.logical_and(a <= Xab, Xab < b)
		phi[idx] = (a * Xsi)[idx]
		idx = np.logical_and(b <= Xab, Xab < c)
		phi[idx] = (a * Xsi * ((c - Xab) / (c - b)))[idx]

		self.mu_gh  = np.mean(phi, 0)
		self.sig_gh = np.std(phi, 0) 
		return self

	def _transform(self, X, *args, **kwargs):
		return 0.5 * (np.tanh(self.scale * ((X - self.mu_gh)/self.sig_gh)) + 1)

	def _inverse_transform(self, X, *args, **kwargs):
		return ((np.tan(X * 2 - 1) / self.scale) * self.sig_gh) + self.mu_gh
	

class TransformerPipeline(CustomTransformer):
	''' Apply multiple transformers seamlessly '''
	
	def __init__(self, scalers=[]):
		self.scalers = scalers

	def _fit(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return self 

	def _transform(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.transform(X, *args, **kwargs)
		return X

	def _inverse_transform(self, X, *args, **kwargs):
		for scaler in self.scalers[::-1]:
			X = scaler.inverse_transform(X, *args, **kwargs)
		return X

	def fit_transform(self, X, *args, **kwargs):
		# Manually apply a fit_transform to avoid transforming twice
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return X


class CustomUnpickler(pkl.Unpickler):
	''' Ensure the classes are found, without requiring an import '''
	_warned = False

	def find_class(self, module, name):
		if name in globals():
			return globals()[name]
		return super().find_class(module, name)

	def load(self, *args, **kwargs):
		with warnings.catch_warnings(record=True) as w:
			pickled_object = super().load(*args, **kwargs)

		# For whatever reason, warnings does not respect the 'once' action for
		# sklearn's "UserWarning: trying to unpickle [...] from version [...] when
		# using version [...]". So instead, we catch it ourselves, and set the 
		# 'once' tracker via the unpickler itself.
		if len(w) and not CustomUnpickler._warned: 
			warnings.warn(w[0].message, w[0].category)
			CustomUnpickler._warned = True 
		return pickled_object


def generate_scalers(args, x_train=None, x_test=None, column_bagging=False):
	''' Add scalers to the args object based on the contained parameter settings '''
	wavelengths  = get_sensor_bands(args.sensor, args)
	store_scaler = lambda scaler, args=[], kwargs={}: (scaler, args, kwargs)
	setattr(args, 'data_wavelengths', wavelengths)

	# Note that the scaler list is applied in order, e.g. MinMaxScaler( LogTransformer(y) )
	args.x_scalers = [
			store_scaler(preprocessing.RobustScaler),
	]
	args.y_scalers = [
		store_scaler(LogTransformer),
		store_scaler(preprocessing.MinMaxScaler, [(-1, 1)]),
	]

	# We only want bagging to be applied to the columns if there are a large number of extra features (e.g. ancillary features included) 
	many_features = column_bagging and any(x is not None and (x.shape[1]-len(wavelengths)) > 15 for x in [x_train, x_test])

	# Add bagging to the columns (use a random subset of columns, excluding the first <n_wavelengths> columns from the process)
	if column_bagging and using_feature(args, 'bagging') and (using_feature(args, 'ratio') or many_features):
		n_extra = 0 if not using_feature(args, 'ratio') else RatioTransformer(wavelengths).get_n_features() # Number of ratio features added
		args.x_scalers = [
			store_scaler(BaggingColumnTransformer, [len(wavelengths)], {'n_extra':n_extra}),
		] + args.x_scalers
	
	# Feature selection via mutual information
	if using_feature(args, 'kbest'):
		args.x_scalers = [
			store_scaler(KBestTransformer, [args.use_kbest]),
		] + args.x_scalers

	# Add additional features to the inputs
	if using_feature(args, 'ratio'):
		kwargs = {}
		if using_feature(args, 'excl_Rrs'):  kwargs.update({'excl_Rrs'    : True})
		if using_feature(args, 'all_ratio'): kwargs.update({'all_ratio' : True})
		args.x_scalers = [
			store_scaler(RatioTransformer, [list(wavelengths)], kwargs),
		] + args.x_scalers

	# Normalize input features using AUC
	if using_feature(args, 'auc'):
		args.x_scalers = [
			store_scaler(AUCTransformer, [list(wavelengths)]),
		] + args.x_scalers
