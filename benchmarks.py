from .metrics          import performance
from .utils            import find_wavelength, ignore_warnings
from .meta             import get_sensor_bands
from .transformers     import TransformerPipeline, LogTransformer
from .Benchmarks.utils import get_benchmark_models

from importlib import import_module
from functools import partial, update_wrapper
from pathlib   import Path 

import numpy as np
import pkgutil, warnings, sys, os


def get_methods(wavelengths, sensor, product, debug=False, allow_opt=False, **kwargs):
	''' Retrieve all benchmark functions from the appropriate product
		directory. Import each function with "model" in the function
		name, ensure any necessary parameters have a default value 
		available, and test whether the function can be run with the
		given wavelengths. A template folder for new algorithms is 
		available in the Benchmarks directory.
	'''
	models = get_benchmark_models(product, allow_opt, debug)
	sensor = sensor.split('-')[0] # remove any extra data which is used by the MDN
	valid  = {}

	# Gather all models which return an output with the available wavelengths
	for name, model in models.items():
		sample_input = np.ones((1, len(wavelengths)))
		model_args   = [sample_input, wavelengths, sensor]
		model_kwargs = dict(kwargs)

		# Add a set of optimization dummy parameters to check if wavelengths are valid for the method
		if getattr(model, 'has_default', False):
			model_kwargs.update( dict(zip(model.opt_vars, [1]*len(model.opt_vars))) )

		try:
			assert(model(*model_args, **model_kwargs) is not None), 'Output is None'
			valid[name] = update_wrapper(partial(model, sensor=sensor), model)
		except Exception as e: 
			if debug: print(f'Exception for function {name}: {e}')		
	return valid


@ignore_warnings
def bench_product(sensor, X, y=None, bands=None, args=None, slices=None, silent=False, product='chl', method=None):	
	if bands is None:
		bands = get_sensor_bands(sensor, args)
	assert(X.shape[1] == len(bands)), f'Too many features given as bands for {sensor}: {X.shape[1]} vs {len(bands)}'

	if product in ['a', 'aph', 'ap', 'ag', 'aph', 'adg', 'b', 'bbp']:
		from .QAA import QAA
		methods = {'QAA': lambda *args, **kwargs: QAA(*args)[product]}
	
		if product in ['a', 'aph', 'adg', 'b', 'bbp']:
			from .GIOP.giop import GIOP
			methods['GIOP'] = lambda *args, **kwargs: GIOP(*args, sensor=sensor)[product]
	else:
		methods = get_methods(bands, sensor, product, tol=15)
	assert(method is None or method in methods), f'Unknown algorithm "{method}". Options are: \n{list(methods.keys())}'

	ests = []
	lbls = []
	for name, func in methods.items():
		if method is None or name == method:
			est_val = func(X, bands, tol=15)
			if product == 'chl':
				# est_val[~np.isfinite(est_val)] = 0
				est_val[est_val < 0] = np.nan#0

			if not silent and y is not None:
				curr_slice = slices
				if slices is None:
					assert(y.shape[1] == est_val.shape[1]), 'Ambiguous y data provided - need to give slices parameter.'
					curr_slice = {product:slice(None)}

				ins_val = y[:, curr_slice[product]]
				for i in range(ins_val.shape[1]):
					print( performance(name, ins_val[:, i], est_val) )
			ests.append(est_val)
			lbls.append(name)
	return dict(zip(lbls, ests))
	

def bench_opt(args, sensor, x_train, x_test, y_train, y_test, slices, silent=False, product='chl'):
	waves   = np.array(get_sensor_bands(sensor))
	methods = get_methods(waves, sensor, product, allow_opt=True)

	ests = []
	lbls = []
	for name, method in methods.items():
		method.fit(x_train, y_train, waves)
		est_chl = method.predict(x_test)

		if not silent:
			ins_val = y_test[:, slices[product]]
			for i in range(ins_val.shape[1]):
				print( performance(name.split('_')[0], ins_val[:, i], est_chl) )
		ests.append(est_chl)
		lbls.append(name+'_opt')
	return dict(zip(lbls, ests))


@ignore_warnings
def bench_ml(args, sensor, x_train, y_train, x_test, y_test, slices=None, silent=False, product='chl', x_other=None, 
			bagging=True, gridsearch=False, scale=True):
	from sklearn.preprocessing import RobustScaler, QuantileTransformer, MinMaxScaler
	from sklearn.model_selection import GridSearchCV
	from sklearn.ensemble import BaggingRegressor
	from sklearn import gaussian_process, svm, neural_network, kernel_ridge, neighbors
	from xgboost import XGBRegressor as XGB
	from sklearn.exceptions import ConvergenceWarning
	warnings.simplefilter("always", ConvergenceWarning)
	assert(y_train.shape[1] == 1), 'Can only use ML benchmarks on chl data'
	from .mdn          import MDN 

	methods = {
		'XGB' : {
			'class'   : XGB,
			'default' : {'max_depth': 15, 'n_estimators': 50, 'objective': 'reg:squarederror'},
			'grid'    : {
				'n_estimators' : [10, 50, 100],
				'max_depth'    : [5, 15, 30],
				'objective'    : ['reg:squarederror'],
			}},
		'SVM' : {
			'class'   : svm.SVR,
			'default' : {'C': 1e1, 'gamma': 'scale', 'kernel': 'rbf'},
			'grid'    : {
				'kernel' : ['rbf', 'poly'],
				'gamma'  : ['auto', 'scale'],
				'C'      : [1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2],
			}},
		'MLP' : {
			'class'   : neural_network.MLPRegressor,
			'default' : {'alpha': 1e-05, 'hidden_layer_sizes': [100]*5, 'learning_rate': 'constant'},
			'grid'    : {
				'hidden_layer_sizes' : [[100]*i for i in range(1, 6)],
				'alpha'              : [1e-5, 1e-4, 1e-3, 1e-2],
				'learning_rate'      : ['constant', 'adaptive'],
			}},
		'KNN' : {
			'class'   : neighbors.KNeighborsRegressor,
			'default' : {'n_neighbors': 5, 'p': 1},
			'grid'    : {
				'n_neighbors' : [3, 5, 10, 20],
				'p'           : [1, 2, 3],
			}},
		# 'MDN' : {
		# 	'class'  : MDN,
		# 	'default': {'no_load': True},
		# 	'grid'   : {
		# 		'hidden': [[100]*i for i in [2,3,5]],
		# 		'l2' : [1e-5,1e-4,1e-3],
		# 		'lr' : [1e-5,1e-4,1e-3],
		# 	}},
		# 'KRR' : {
		# 	'class'   : kernel_ridge.KernelRidge,
		# 	'default' : {'alpha': 1e1, 'kernel': 'laplacian'},
		# 	'grid'    : {
		# 		'alpha' : [1e-1, 1e0, 1e1, 1e2],
		# 		'kernel': ['rbf', 'laplacian', 'linear'],
		# 	}},
		# 'GPR' : {
		# 	'class'   : gaussian_process.GaussianProcessRegressor,
		# 	'default' : {'kernel': gaussian_process.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gaussian_process.kernels.RBF(1.0, (1e-1, 1e3))},
		# 	'grid'    : {
		# 		'kernel' : [gaussian_process.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gaussian_process.kernels.RBF(1.0, (1e-1, 1e3)), 
		# 					gaussian_process.kernels.ConstantKernel(10.0, (1e-1, 1e3)) * gaussian_process.kernels.RBF(10.0, (1e-1, 1e3))],
		# 	}},
	}

	# Filter MLP convergence warnings
	if not sys.warnoptions:
		warnings.simplefilter('ignore')
		os.environ['PYTHONWARNINGS'] = 'ignore'

	if False:
		x_train = x_train[:,:len(get_sensor_bands(sensor, args))]
		x_test  = x_test[:,:len(get_sensor_bands(sensor, args))]

	if scale:
		x_scaler = RobustScaler()
		y_scaler = TransformerPipeline([LogTransformer(), MinMaxScaler((-1, 1))]) 
		x_scaler.fit(x_train)
		y_scaler.fit(y_train)
		x_train = x_scaler.transform(x_train)
		x_test  = x_scaler.transform(x_test)
		y_train = y_scaler.transform(y_train)

	if slices is None:
		assert(y_test.shape[1] == y_train.shape[1]), 'Ambiguous y data provided - need to give slices parameter.'
		slices = {product:slice(None)}

	if gridsearch:
		print('\nPerforming gridsearch...')

	other = []
	ests  = []
	lbls  = []
	for method, params in methods.items():
		if gridsearch and method != 'SVM':
			model = GridSearchCV(params['class'](), params['grid'], refit=False, n_jobs=3 if method != 'MDN' else 1, scoring='neg_median_absolute_error')
			model.fit(x_train.copy(), y_train.copy().flatten())

			print(f'Best {method} params: {model.best_params_}')
			model = params['class'](**model.best_params_)

		else:
			model = params['class'](**params['default'])

		if bagging:
			model = BaggingRegressor(model, n_estimators=10, max_samples=0.75, bootstrap=False)
		model.fit(x_train.copy(), y_train.copy().flatten())
		est_val = model.predict(x_test.copy()).flatten()

		if scale:
			est_val = y_scaler.inverse_transform(est_val[:,None]).flatten()	

		if not silent:
			ins_val = y_test[:, slices[product]]
			for i in range(ins_val.shape[1]):
				print( performance(method, ins_val[:, i], est_val) )

		ests.append(est_val[:,None])
		lbls.append(method)

		if x_other is not None:
			est = model.predict(x_other.copy())
			if scale: est = y_scaler.inverse_transform(est[:,None]).flatten()
			other.append(est)

	if not len(other):
		return dict(zip(lbls, ests))
	return dict(zip(lbls, ests)), dict(zip(lbls, other))


def run_benchmarks(args, sensor, x_test, y_test, slices, silent=True, x_train=None, y_train=None, gridsearch=False, with_ml=False):
	benchmarks = {}
	for k in slices:
		if k in ['chl', 'tss', 'a', 'aph', 'ap', 'ag', 'aph', 'adg', 'b', 'bbp']:
			benchmarks.update( bench_product(sensor, x_test, y=y_test, slices=slices, silent=silent, args=args, product=k) )

		if k in ['chl', 'tss'] and with_ml:
			benchmarks.update( bench_ml(args, sensor, x_train, y_train, x_test, y_test, slices, silent, product=k, gridsearch=gridsearch) )
	return benchmarks


def print_benchmarks(*args, **kwargs):
	run_benchmarks(*args, silent=False, **kwargs)
