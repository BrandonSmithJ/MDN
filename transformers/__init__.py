from sklearn import preprocessing

from ..utils import using_feature
from ..meta  import get_sensor_bands 

from ._CustomTransformer           import _CustomTransformer
from .AUCTransformer               import AUCTransformer
from .BaggingColumnTransformer     import BaggingColumnTransformer
from .ColumnSelectionTransformer   import ColumnSelectionTransformer 
from .DatasetMembershipTransformer import DatasetMembershipTransformer
from .ExclusionTransformer         import ExclusionTransformer
from .IdentityTransformer          import IdentityTransformer
from .KBestTransformer             import KBestTransformer
from .LogTransformer               import LogTransformer
from .NegLogTransformer            import NegLogTransformer
from .RatioTransformer             import RatioTransformer
from .TanhTransformer              import TanhTransformer


def generate_scalers(args, x_train=None, x_test=None, column_bagging=False):
	''' Add scalers to the args object based on the contained parameter settings '''
	wavelengths = get_sensor_bands(args.sensor, args)
	serialize   = lambda scaler, args=[], kwargs={}: (scaler, args, kwargs)
	setattr(args, 'wavelengths', wavelengths)

	# Note that the scaler list is applied in order, e.g. MinMaxScaler( LogTransformer(y) )
	args.x_scalers = [
			serialize(preprocessing.RobustScaler),
	]
	args.y_scalers = [
		serialize(LogTransformer),
		serialize(preprocessing.MinMaxScaler, [(-1, 1)]),
	]

	# We only want bagging to be applied to the columns if there are a large number of extra features (e.g. ancillary features included) 
	many_features = column_bagging and any(x is not None and (x.shape[1]-len(wavelengths)) > 15 for x in [x_train, x_test])

	# Add bagging to the columns (use a random subset of columns, excluding the first <n_wavelengths> columns from the process)
	if column_bagging and using_feature(args, 'bagging') and (using_feature(args, 'ratio') or many_features):
		n_extra = 0 if not using_feature(args, 'ratio') else RatioTransformer(wavelengths).get_n_features() # Number of ratio features added
		args.x_scalers = [
			serialize(BaggingColumnTransformer, [len(wavelengths)], {'n_extra':n_extra, 'seed': args.seed}),
		] + args.x_scalers
	
	# Feature selection via mutual information
	if using_feature(args, 'kbest'):
		args.x_scalers = [
			serialize(KBestTransformer, [args.use_kbest]),
		] + args.x_scalers

	# Add additional features to the inputs
	if using_feature(args, 'ratio'):
		kwargs = {}
		if using_feature(args, 'excl_Rrs'):  kwargs.update({'excl_Rrs'    : True})
		if using_feature(args, 'all_ratio'): kwargs.update({'all_ratio' : True})
		args.x_scalers = [
			serialize(RatioTransformer, [list(wavelengths)], kwargs),
		] + args.x_scalers

	# Normalize input features using AUC
	if using_feature(args, 'auc'):
		args.x_scalers = [
			serialize(AUCTransformer, [list(wavelengths)]),
		] + args.x_scalers



class TransformerPipeline(_CustomTransformer):
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





