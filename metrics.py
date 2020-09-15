from .utils import ignore_warnings
from scipy import stats
import numpy as np 
import functools 


def validate_shape(func):
	''' Decorator to flatten all function input arrays, and ensure shapes are the same '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		flat_args = [a.flatten() if hasattr(a, 'flatten') else a for a in args]
		shapes    = [a.shape for a in flat_args if hasattr(a, 'shape')]
		original  = [a.shape for a in args      if hasattr(a, 'shape')]
		assert(all(shapes[0] == s for s in shapes)), f'Shapes mismatch in {func.__name__}: {original}'
		return func(*flat_args, **kwargs)
	return helper  


def only_finite(func):
	''' Decorator to remove samples which are nan in any input array '''
	@validate_shape
	@functools.wraps(func)
	def helper(*args, **kwargs):
		stacked = np.vstack(args)
		valid   = np.all(np.isfinite(stacked), 0)
		assert(valid.sum()), f'No valid samples exist for {func.__name__} metric'
		return func(*stacked[:, valid], **kwargs)
	return helper 


def only_positive(func):
	''' Decorator to remove samples which are zero/negative in any input array '''
	@validate_shape
	@functools.wraps(func)	
	def helper(*args, **kwargs):
		stacked = np.vstack(args)
		valid   = np.all(stacked > 0, 0)
		assert(valid.sum()), f'No valid samples exist for {func.__name__} metric'
		return func(*stacked[:, valid], **kwargs)
	return helper 


def label(name):
	''' Label a function to aid in printing '''
	def wrapper(func):
		func.__name__ = name
		return ignore_warnings(func)
	return wrapper


# ============================================================================
''' 
When executing a function, decorator order starts with the 
outermost decorator and works its way down the stack; e.g.
	@dec1
	@dec2
	def foo(): pass 
	def bar(): pass
And then foo == dec1(dec2(bar)). So, foo will execute dec1, 
then dec2, then the original function. 

Below, in rmsle (for example), we have:
	rmsle = only_finite( only_positive( label(rmsle) ) ) 
This means only_positive() will get the input arrays only
after only_finite() removes any nan samples. As well, both
only_positive() and only_finite() will have access to the 
function __name__ assigned by label().

For all functions below, y=true and y_hat=estimate
'''


@only_finite
@label('RMSE')
def rmse(y, y_hat):
	''' Root Mean Squared Error '''
	return np.mean((y - y_hat) ** 2) ** .5


@only_finite
@only_positive
@label('RMSLE')
def rmsle(y, y_hat):
	''' Root Mean Squared Logarithmic Error '''
	return np.mean(np.abs(np.log(y) - np.log(y_hat)) ** 2) ** 0.5 


@only_finite
@label('NRMSE')
def nrmse(y, y_hat):
	''' Normalized Root Mean Squared Error '''
	return ((y - y_hat) ** 2).mean() ** .5 / y.mean()


@only_finite
@label('MAE')
def mae(y, y_hat):
	''' Mean Absolute Error '''
	return np.mean(np.abs(y - y_hat))


@only_finite
@label('MAPE')
def mape(y, y_hat):
	''' Mean Absolute Percentage Error '''
	return 100 * np.mean(np.abs((y - y_hat) / y))


@only_finite
@label('<=0')
def leqz(y, y_hat=None):
	''' Less than or equal to zero (y_hat) '''
	if y_hat is None: y_hat = y
	return (y_hat <= 0).sum()


@validate_shape
@label('<=0|NaN')
def leqznan(y, y_hat=None):
	''' Less than or equal to zero (y_hat) '''
	if y_hat is None: y_hat = y
	return np.logical_or(np.isnan(y_hat), y_hat <= 0).sum()


@only_finite
@only_positive
@label('MdSA')
def mdsa(y, y_hat):
	''' Median Symmetric Accuracy '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	return 100 * (np.exp(np.median(np.abs(np.log(y_hat / y)))) - 1)


@only_finite
@only_positive
@label('MSA')
def msa(y, y_hat):
	''' Mean Symmetric Accuracy '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	return 100 * (np.exp(np.mean(np.abs(np.log(y_hat / y)))) - 1)


@only_finite
@only_positive
@label('SSPB')
def sspb(y, y_hat):
	''' Symmetric Signed Percentage Bias '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	M = np.median( np.log(y_hat / y) )
	return 100 * np.sign(M) * (np.exp(np.abs(M)) - 1)


@only_finite
@label('Bias')
def bias(y, y_hat):
	''' Mean Bias '''
	return np.mean(y_hat - y)


@only_finite
@only_positive
@label('R^2')
def r_squared(y, y_hat):
	''' Logarithmic R^2 '''
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(np.log10(y), np.log10(y_hat))
	return r_value**2


@only_finite
@only_positive
@label('Slope')
def slope(y, y_hat):
	''' Logarithmic slope '''
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(np.log10(y), np.log10(y_hat))
	return slope_


@only_finite
@only_positive
@label('Intercept')
def intercept(y, y_hat):
	''' Locarithmic intercept '''
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(np.log10(y), np.log10(y_hat))
	return intercept_


@validate_shape
@label('MWR')
def mwr(y, y_hat, y_bench):
	''' 
	Model Win Rate - Percent of samples in which model has a closer 
	estimate than the benchmark.
		y: true, y_hat: model, y_bench: benchmark 
	'''
	y_bench[y_bench < 0] = np.nan 
	y_hat[y_hat < 0] = np.nan 
	y[y < 0] = np.nan 
	valid = np.logical_and(np.isfinite(y_hat), np.isfinite(y_bench))
	diff1 = np.abs(y[valid] - y_hat[valid])
	diff2 = np.abs(y[valid] - y_bench[valid])
	stats = np.zeros(len(y))
	stats[valid]  = diff1 < diff2
	stats[~np.isfinite(y_bench)] = 1
	stats[~np.isfinite(y_hat)] = 0
	return stats.sum() / np.isfinite(y).sum()


def performance(key, y, y_hat, metrics=[mdsa, sspb, slope, rmse, rmsle, mae, leqznan]):
	''' Return a string containing performance using various metrics. 
		y should be the true value, y_hat the estimated value. '''
	return '%8s | %s' % (key, '   '.join([
			'%s: %6.3f' % (f.__name__, f(y,y_hat)) for f in metrics]))