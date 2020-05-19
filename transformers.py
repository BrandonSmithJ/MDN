from sklearn.base import TransformerMixin
from sklearn import preprocessing
import pickle as pkl
import numpy as np 


class CustomTransformer(TransformerMixin):
	''' Data transformer class which validates data shapes. 
		Child classes should override _fit, _transform, _inverse_transform '''
	_input_shape  = None 
	_output_shape = None

	def fit(self, X, *args, **kwargs):				 
		self._input_shape = X.shape[1]
		return self._fit(X, *args, **kwargs)

	def transform(self, X, *args, **kwargs):
		if self._input_shape is not None:
			assert(X.shape[1] == self._input_shape), f'Number of data features changed: {self._input_shape} vs {X.shape[1]}'
		X = self._transform(X, *args, **kwargs)
		
		if self._output_shape is not None:
			assert(X.shape[1] == self._output_shape), f'Number of data features changed: {self._output_shape} vs {X.shape[1]}'
		self._output_shape = X.shape[1]
		return X 

	def inverse_transform(self, X, *args, **kwargs):
		if self._output_shape is not None:
			assert(X.shape[1] == self._output_shape), f'Number of data features changed: {self._output_shape} vs {X.shape[1]}'
		X = self._inverse_transform(X, *args, **kwargs)
		
		if self._input_shape is not None:
			assert(X.shape[1] == self._input_shape), f'Number of data features changed: {self._input_shape} vs {X.shape[1]}'
		self._input_shape = X.shape[1]
		return X 

	def _fit(self, X, *args, **kwargs):				  return self
	def _transform(self, X, *args, **kwargs):		  raise NotImplemented
	def _inverse_transform(self, X, *args, **kwargs): raise NotImplemented



class IdentityTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):		 return X
	def _inverse_transform(self, X, *args, **kwargs): return X


class LogTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):		 return np.log(X)
	def _inverse_transform(self, X, *args, **kwargs): return np.exp(X)


class NegLogTransformer(CustomTransformer):
	''' 
	Log-like transformation which allows negative values (Whittaker et al. 2005)
	http://fmwww.bc.edu/repec/bocode/t/transint.html
	'''
	def _transform(self, X, *args, **kwargs):		 return np.sign(X) *  np.log(np.abs(X)  + 1)
	def _inverse_transform(self, X, *args, **kwargs): return np.sign(X) * (np.exp(np.abs(X)) - 1)


class ColumnTransformer(CustomTransformer):
	''' Reduce columns to specified selections (feature selection) '''
	def __init__(self, columns, *args, **kwargs):	 self._c = columns 
	def _transform(self, X, *args, **kwargs):		 return X[:, self._c]


class BaggingColumnTransformer(CustomTransformer):
	''' Randomly select a percentage of columns to drop '''
	percent = 1#0.75

	def __init__(self, n_bands, *args, **kwargs):
		self.n_bands = n_bands

	def _fit(self, X, *args, **kwargs):
		shp  = X.shape[1] - self.n_bands
		ncol = int(shp*self.percent)
		cols = np.arange(shp) + self.n_bands
		np.random.shuffle(cols)
		self.cols = np.append(np.arange(self.n_bands), cols[:ncol], 0)
		print(f'Reducing bands from {shp} ({X.shape[1]} total) to {ncol} ({len(self.cols)} total) ({self.cols})')
		return self

	def _transform(self, X, *args, **kwargs):
		return X[:, self.cols]


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

	def _fit(self, X):
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
	def __init__(self, wavelengths, *args, **kwargs):
		self.wavelengths = wavelengths

	def _fit(self, X):
		self.shape = X.shape[1]
		return self 

	def _transform(self, X, *args, **kwargs):		 
		''' 
		Simple feature engineering method. Add band 
		ratios as features. Does not add reciprocal 
		ratios or any other duplicate features; 
		adds a band sum ratio (based on	three-band 
		Chl retrieval method).

		Usage:
			# one sample with three features, shaped [samples, features]
			x = [[a, b, c]] 
			y = ratio(x)
				# y -> [[a, b, c, b/a, b/(a+c), c/a, c/(a+b), c/b, a/(b+c)]
		'''
		x     = np.atleast_2d(X)
		x_new = [v for v in x.T]

		# Band ratios
		if len(self.wavelengths) < 6:
			for i, L1 in enumerate(self.wavelengths):
				for j, L2 in enumerate(self.wavelengths):
					if L1 < L2:
						R1 = x[:, i]
						R2 = x[:, j] 
						# if len(self.wavelengths) < 6 or (L1 > 500 and L2 > 500):
						x_new.append(R2 / R1)
						print(f'{L2}/{L1}', np.min(x_new[-1]), np.max(x_new[-1]))

						for k, L3 in enumerate(self.wavelengths):
							if L3 not in [L1, L2]:
								R3 = x[:, k]
								if len(self.wavelengths) < 6 or L3 > 500:
									if L1 < L3:
										x_new.append(R2 / (R1 + R3))
									else:
										x_new.append(R3 / (R1 + R2))
		

		# Line height variations, examining height of center between two shoulder bands
		for i, L1 in enumerate(self.wavelengths):
			for j, L2 in enumerate(self.wavelengths):
				for k, L3 in enumerate(self.wavelengths):
					if (L3 > L2) and (L2 > L1) and L2 < 900:
						if len(self.wavelengths) < 6:# or not (L1 == 655 and L2 == 865 and L3 == 1609): #L1 > 600:# and L2  600 and L3 > 600):
							c  = (L3 - L2) / (L3 - L1)
							R1 = x[:, i]
							R2 = x[:, j]
							R3 = x[:, k]
							x_new.append(R2 - R3 - c*(R1-R3))
							print(f'{len(self.wavelengths)} ({L3}-{L2})/({L3}-{L1})', np.min(x_new[-1]), np.max(x_new[-1]))

		return np.hstack([v[:,None] for v in x_new])

	def _inverse_transform(self, X, *args, **kwargs): 
		return np.array(X)[:, :self.shape]



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
	
	def __init__(self, scalers=None):
		if scalers is None or len(scalers) == 0: 	
			self.scalers = [
				# NegLogTransformer(), 
				LogTransformer(),
				preprocessing.RobustScaler(),
				preprocessing.MinMaxScaler((1e-2, 1)),
			]
		else:
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



class CustomUnpickler(pkl.Unpickler):
	''' Ensure the classes are found, without requiring an import '''
	def find_class(self, module, name):
		if name in globals():
			return globals()[name]
		return super().find_class(module, name)