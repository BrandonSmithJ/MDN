from scipy import stats
import numpy as np 
import functools 
import warnings 

def flatten(func):
	''' Decorator to flatten function parameters '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		flat_args = [a if a is None else a.flatten() for a in args]
		return func(*flat_args, **kwargs)
	return helper 


def only_valid(func):
	''' Decorator to remove all elements having a nan in any array '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		assert(all([len(a.shape) == 1 for a in args]))
		stacked = np.vstack(args)
		valid   = np.all(np.isfinite(stacked), 0)
		return func(*stacked[:, valid], **kwargs)
	return helper 


def only_positive(func):
	''' Decorator to remove all elements having a zero/negative value in any array '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		assert(all([len(a.shape) == 1 for a in args]))
		stacked = np.vstack(args)
		valid   = np.all(stacked > 0, 0)
		return func(*stacked[:, valid], **kwargs)
	return helper 


def label(name):
	''' Label a function for when it's printed '''
	def helper(f):
		f.__name__ = name
		return f
	return helper


@label('RMSE')
@flatten
@only_valid
def rmse(y1, y2):
	''' Root Mean Squared Error '''
	# return ((y1 - y2) ** 2).mean() ** .5
	return np.mean((y1 - y2) ** 2) ** .5


@label('RMSLE')
@flatten
@only_valid
@only_positive
def rmsle(y1, y2):
	''' Root Mean Squared Logarithmic Error '''
	return np.mean(np.abs(np.log(y1) - np.log(y2)) ** 2) ** 0.5 


@label('NRMSE')
@flatten
@only_valid
def nrmse(y1, y2):
	''' Normalized Root Mean Squared Error '''
	return ((y1 - y2) ** 2).mean() ** .5 / y1.mean()


@label('MAE')
@flatten
@only_valid
def mae(y1, y2):
	''' Mean Absolute Error '''
	return np.mean(np.abs(y1 - y2))


# @label('MAE')
# @flatten
# @only_valid
# def mae(y1, y2):
# 	''' Mean Absolute Error '''
# 	i  = np.logical_and(y1 > 0, y2 > 0)
# 	y1 = np.log10(y1[i])
# 	y2 = np.log10(y2[i])
# 	i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
# 	y1 = y1[i]
# 	y2 = y2[i]
# 	return 10**np.median(np.abs(y1 - y2))


@label('MAPE')
@flatten
@only_valid
def mape(y1, y2):
	''' Mean Absolute Percentage Error '''
	return 100 * np.median(np.abs((y1 - y2) / y1))


@label('<=0')
@flatten
@only_valid
def leqz(y1, y2=None):
	''' Less than or equal to zero (y2) '''
	if y2 is None: y2 = y1
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		return (y2 <= 0).sum()


@label('<=0|NaN')
@flatten
def leqznan(y1, y2=None):
	''' Less than or equal to zero (y2) '''
	if y2 is None: y2 = y1
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		return np.logical_or(np.isnan(y2), y2 <= 0).sum()


@label('MdSA')
@flatten
@only_valid
@only_positive
def mdsa(y1, y2):
	''' Median Symmetric Accuracy '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	Q = np.log(y2 / y1)
	i = np.isfinite(Q)
	return 100 * (np.exp(np.median(np.abs(Q[i]))) - 1)


@label('MSA')
@flatten
@only_valid
@only_positive
def msa(y1, y2):
	''' Mean Symmetric Accuracy '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	Q = np.log(y2 / y1)
	i = np.isfinite(Q)
	return 100 * (np.exp(np.mean(np.abs(Q[i]))) - 1)



@label('SSPB')
@flatten
@only_valid
@only_positive
def sspb(y1, y2):
	''' Symmetric Signed Percentage Bias '''
	# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017SW001669
	Q = np.log(y2 / y1)
	i = np.isfinite(Q)
	M = np.median(Q[i])
	return 100 * np.sign(M) * (np.exp(np.abs(M)) - 1)


@label('Bias')
@flatten
@only_valid
def bias(y1, y2):
	''' Mean Bias '''
	return np.mean(y2 - y1)


# @label('Bias')
# @flatten
# @only_valid
# def bias(y1, y2):
# 	''' Median Bias '''
# 	# return np.median(y2 - y1)
# 	i  = np.logical_and(y1 > 0, y2 > 0)
# 	y1 = np.log10(y1[i])
# 	y2 = np.log10(y2[i])
# 	i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
# 	y1 = y1[i]
# 	y2 = y2[i]
# 	return 10**np.median(y2 - y1)


@label('R^2')
@flatten
@only_valid
@only_positive
def r_squared(y1, y2):
	y1 = np.log10(y1)
	y2 = np.log10(y2)
	i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
	if i.sum() < 3: return np.nan
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(y1[i],y2[i])
	return r_value**2


@label('Slope')
@flatten
@only_valid
@only_positive
def slope(y1, y2):
	y1 = np.log10(y1)
	y2 = np.log10(y2)
	i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
	if i.sum() < 3: return np.nan
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(y1[i],y2[i])
	return slope_


@label('Intercept')
@flatten
@only_valid
@only_positive
def intercept(y1, y2):
	y1 = np.log10(y1[i])
	y2 = np.log10(y2[i])
	i  = np.logical_and(np.isfinite(y1), np.isfinite(y2))
	if i.sum() < 3: return np.nan
	slope_, intercept_, r_value, p_value, std_err = stats.linregress(y1[i],y2[i])
	return intercept_


@label('MWR')
@flatten
def mwr(y1, y2, y3):
	''' Model win rate - y1: true, y2: model, y3: benchmark '''
	y3[y3 < 0] = np.nan 
	y2[y2 < 0] = np.nan 
	y1[y1 < 0] = np.nan 
	valid = np.logical_and(np.isfinite(y2), np.isfinite(y3))
	diff1 = np.abs(y1[valid] - y2[valid])
	diff2 = np.abs(y1[valid] - y3[valid])
	stats = np.zeros(len(y1))
	stats[valid]  = diff1 < diff2
	stats[~np.isfinite(y3)] = 1
	stats[~np.isfinite(y2)] = 0
	return stats.sum() / np.isfinite(y1).sum()


def performance(key, y1, y2, metrics=[rmse, slope, mdsa, rmsle, sspb, mape, mae, bias, leqznan]):#[rmse, rmsle, mape, r_squared, bias, mae, leqznan, slope]):
	''' Return a string containing performance using various metrics. 
		y1 should be the true value, y2 the estimated value. '''
	return '%8s | %s' % (key, '   '.join([
			'%s: %6.3f' % (f.__name__, f(y1,y2)) for f in metrics]))