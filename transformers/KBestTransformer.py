from ._CustomTransformer import _CustomTransformer

from sklearn.feature_selection import SelectKBest, mutual_info_regression
import numpy as np 


class KBestTransformer(_CustomTransformer):
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
