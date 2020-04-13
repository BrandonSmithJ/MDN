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



class RatioTransformer(CustomTransformer):	
	''' Add ratio features '''
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
		if len(x_new) > 20:
			keep  = np.array([False] * x.shape[1])
			keep[::5] = True
			keep[-1]  = True
			print(f'Too many features ({x.shape[1]}) - using only a few bands ({keep.sum()})')
			from QAA import wavelengths, find
			# keep  = np.array([False] * x.shape[1])
			# for i in wavelengths['OLCI']:
			# 	keep[find(i, np.array(wavelengths['HICO']))] = True 

			x_new = ratio(x[:, keep])[:, keep.sum():]
			x_new = np.append(x, x_new, 1)

		else:
			# from QAA import wavelengths, find
			# wavelengths = np.array(wavelengths['S2B'])
			# b705 = find(705, wavelengths)
			# b655 = find(560, wavelengths)
			# b560 = find(560, wavelengths)
			# b490 = find(443, wavelengths)
			# assert(0), [x.max(0), x.min(0)]
			# [443, 490, 560, 665, 705, 740, 783],
			# x_new.append(x[:, b705] / x[:,b655])
			# x_new.append(x[:, b560] / x[:,b490])
			# rmax = .3
			# rmin = 1e-5
			# x = x.copy()
			# # x[x > rmax] = rmax 
			# # x[x < rmin] = rmin

			# x[np.any(x > rmax, 1)] += -x[np.any(x > rmax, 1)].max(1, keepdims=True) + rmax
			# x[np.any(x < rmin, 1)] += -x[np.any(x < rmin, 1)].min(1, keepdims=True) + rmin
			# x_new = [v for v in x.T]
			# x = x.round(4)
			# x = (x*2500).astype(np.int32).astype(np.float32)/2500.
			# x[x==0] = 1/2500.
			for i in range(x.shape[1]):
				for j in range(x.shape[1]):
					x = x.copy() * 1e3
					if i < j: 
						x_new.append(x[:,j] * x[:,i])
						for k in range(x.shape[1]):
							if k != i and k != j:
								if i < k:
									x_new.append(x[:,j] * (x[:,i] + x[:,k]))
								else:
									x_new.append(x[:,k] * (x[:,i] + x[:,j]))
			
			def FLH():
				from QAA import wavelengths, find
				wvls = np.array(wavelengths['MODA'])
				w709 = x[:, find(709, wvls)]
				w681 = x[:, find(681, wvls)]
				w665 = x[:, find(655, wvls)]
				return w681 - 1.005 * (w665 + (w709 - w665) * ((681-665)/(709-665)))
			
			def MCI():
				from QAA import wavelengths, find
				wvls = np.array(wavelengths['MODA'])
				w753 = x[:, find(753, wvls)]
				w709 = x[:, find(709, wvls)]
				w680 = x[:, find(680, wvls)]
				return w709 - 1.005 * (w680 + (w753 - w680) * ((709-680)/(753-680)))
			# x_new.append(FLH())
			# x_new.append(MCI())


			x_new = np.hstack([v[:,None] for v in x_new])
			#if x.shape[1] > 20:
				#print(f'Too many features ({x_new.shape[1]}) - using PCA')
				#valid = np.all(np.logical_and(np.isfinite(x_new), x_new < 1e4), 0)
				#x_new = x_new[:,valid]
				#from sklearn.decomposition import FastICA as PCA
				#p = PCA(whiten=True)
				#x_new = p.fit_transform(x_new)[:,:x.shape[1]]
		return x_new

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