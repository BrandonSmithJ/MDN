from functools import partial
from scipy.optimize import minimize
from functools import update_wrapper, partial
import numpy as np 
import warnings


def find_band_idx(w, wavelengths):
	# Index of closest wavelength
	wavelengths = np.array(wavelengths)
	return np.abs(wavelengths - np.atleast_1d(w)[:,None]).argmin(1) 	


def closest_band(w, wavelengths):
	# Value of closest wavelength
	wavelengths = np.array(wavelengths)
	return wavelengths[find_band_idx(w, wavelengths)]


def has_band(w, wavelengths, tol=5):
	# Closest band within <tol> nm
	wavelengths = np.array(wavelengths)
	return np.abs(w - closest_band(w, wavelengths)) <= tol


def to_rrs(Rrs):
	# Conversion to subsurface reflectance (Lee et al. 2002)
	return Rrs / (0.52 + 1.7 * Rrs)


def to_Rrs(rrs):
	# Inverse of to_rrs - conversion from subsurface to remote sensing reflectance
	return (rrs * 0.52) / (1 - rrs * 1.7)


def get_required(Rrs, wavelengths, required=[], tol=5):
	''' 
	Checks that all required wavelengths are available in the given data. 
	Returns an object which acts as a functional interface into the Rrs data,
	allowing a wavelength or set of wavelengths to be returned:
		Rrs = get_required(Rrs, ...)
		Rrs(443)        # Returns a matrix containing the band data closest to 443nm (shape [N, 1])
		Rrs([440, 740]) # Returns a matrix containing the band data closest to 440nm, and to 740nm (shape [N, 2])
	'''
	wavelengths = np.array(wavelengths)
	Rrs = np.atleast_2d(Rrs)
	assert(Rrs.shape[1] == len(wavelengths)), \
		f'Shape mismatch: Rrs={Rrs.shape}, wavelengths={len(wavelengths)}'
	assert(all([has_band(w, wavelengths, tol) for w in required])), \
		f'At least one of {required} is missing from {wavelengths}'
	return lambda w: Rrs[:, find_band_idx(w, wavelengths)] if w is not None else Rrs


class Optimizer:
	def __init__(self, function, opt_vars, has_default):
		self.function    = function
		self.opt_vars    = opt_vars
		self.has_default = has_default

	def __call__(self, *args, **kwargs):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore')
			return self.function(*args, **kwargs)

	def fit(self, X, Y, wavelengths):
		def cost_func(guess):
			assert(np.all(np.isfinite(guess))), guess
			guess = dict(zip(self.opt_vars, guess))
			return np.nanmedian(np.abs((self(X, wavelengths, **guess) - Y) / Y))
			return np.abs(np.nanmean(self(X, wavelengths, **guess) - Y))
			return ((self(X, wavelengths, **guess) - Y) ** 2).sum() ** 0.5
		from skopt import gbrt_minimize
		init = [(1e-2,100)]*len(self.opt_vars)
		# res  = minimize(cost_func, init, tol=1e-6, options={'maxiter':1e3}, method='BFGS')
		res  = gbrt_minimize(cost_func, init, n_random_starts=10000, n_calls=10000)#, method='SLSQP')#, tol=1e-10, options={'maxiter':1e5}, method='SLSQP')
		print(self.__name__, res.x, res.fun)

		self.trained_function = partial(self.function, wavelengths=wavelengths, **dict(zip(self.opt_vars, res.x)))

	def predict(self, X):
		return self.trained_function(X)


def optimize(opt_vars, has_default=True):
	''' Can automatically optimize a function 
		with a given set of variables, using the
		first set of data given. Then, return the 
		optimized function as partially defined, using
		the optimal parameters
	''' 

	def function_wrapper(function):
		return Optimizer(function, opt_vars, has_default)
	return function_wrapper


# def set_name(name, extra_kws={}):
# 	''' Set the model name to be different than the containing folder '''
# 	def function_wrapper(function):
# 		function.model_name = name
# 		if hasattr(function, 'function'):
# 			new_function      = partial(function.function, **extra_kws)
# 			function.function = update_wrapper(new_function, function.function)
# 		else:
# 			new_function = partial(function, **extra_kws)
# 			function     = update_wrapper(new_function, function)
# 		return function
# 	return function_wrapper