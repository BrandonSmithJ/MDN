from xgboost import XGBRegressor as XGB
from sklearn.svm import SVR as SVR_sklearn
from sklearn.neighbors import KNeighborsRegressor as KNN_sklearn
from sklearn.neural_network import MLPRegressor as MLP 
from sklearn.linear_model import BayesianRidge as BRR_sklearn
from sklearn.kernel_ridge import KernelRidge as KRR 
from sklearn.gaussian_process import GaussianProcessRegressor as GPR, kernels as GPK
# from ...mdn2 import MDN


class Sklearn_deterministic:
	''' Class which allow a random_state kwarg to be passed in, and set,
		via the model constructor - ensuring we have a uniform interface for 
		the models. For instance, comments suggest SVR is already deterministic,
		but the random_state kwarg is not exposed in the constructor: therefore
		we expose the random_state parameter, which was previously set to None 
		in the BaseLibSVM super constructor. '''

	def __init__(self, random_state=None, **kwargs):
		super().__init__(**kwargs)
		self.random_state = random_state


class SVR(Sklearn_deterministic, SVR_sklearn): pass
class KNN(Sklearn_deterministic, KNN_sklearn): pass
class BRR(Sklearn_deterministic, BRR_sklearn): pass


models = {
	'XGB' : {
		'class'   : XGB,
		'default' : {'max_depth': 15, 'n_estimators': 50, 'objective': 'reg:squarederror'},
		'grid'    : {
			'n_estimators' : [10, 50, 100],
			'max_depth'    : [5, 15, 30],
			'objective'    : ['reg:squarederror'],
	}},

	'SVM' : {
		'class'   : SVR,
		'default' : {'C': 1e1, 'gamma': 'scale', 'kernel': 'rbf'},
		'grid'    : {
			'kernel' : ['rbf', 'poly'],
			'gamma'  : ['auto', 'scale'],
			'C'      : [1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2],
	}},

	'MLP' : {
		'class'   : MLP,
		'default' : {'alpha': 1e-05, 'hidden_layer_sizes': [100]*5, 'learning_rate': 'constant'},
		'grid'    : {
			'hidden_layer_sizes' : [[100]*i for i in range(1, 6)],
			'alpha'              : [1e-5, 1e-4, 1e-3, 1e-2],
			'learning_rate'      : ['constant', 'adaptive'],
	}},

	'KNN' : {
		'class'   : KNN,
		'default' : {'n_neighbors': 5, 'p': 1},
		'grid'    : {
			'n_neighbors' : [3, 5, 10, 20],
			'p'           : [1, 2, 3],
	}},

	'BRR' : {
		'class'    : BRR, 
		'default'  : {'normalize': True},
		'grid'     : {

	}},

	# 'KRR' : {
	# 	'class'   : KRR,
	# 	'default' : {'alpha': 1e1, 'kernel': 'laplacian'},
	# 	'grid'    : {
	# 		'alpha' : [1e-1, 1e0, 1e1, 1e2],
	# 		'kernel': ['rbf', 'laplacian', 'linear'],
	# }},

	'GPR' : {
		'class'   : GPR,
		'default' : {'normalize_y': True, 'alpha': 1e-3},#'kernel': GPK.ConstantKernel(1.0, (1e-1, 1e3)) * GPK.RBF(1.0, (1e-1, 1e3))},
		'grid'    : {
			'kernel' : [GPK.ConstantKernel(1.0, (1e-1, 1e3)) * GPK.RBF(1.0, (1e-1, 1e3)), 
						GPK.ConstantKernel(10.0, (1e-1, 1e3)) * GPK.RBF(10.0, (1e-1, 1e3))],
	}},

	# 'MDN' : {
	# 	'class'  : MDN,
	# 	'default': {'no_load': True},
	# 	'grid'   : {
	# 		'hidden': [[100]*i for i in [2,3,5]],
	# 		'l2' : [1e-5,1e-4,1e-3],
	# 		'lr' : [1e-5,1e-4,1e-3],
	# }},

}