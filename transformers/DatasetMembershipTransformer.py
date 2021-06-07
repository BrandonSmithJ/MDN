from ._CustomTransformer import _CustomTransformer

from sklearn import preprocessing
import numpy as np 


class DatasetMembershipTransformer(_CustomTransformer):
	''' Appends a one-hot vector of features to each sample, indicating dataset membership  '''

	def __init__(self, datasets, *args, **kwargs):
		self.locs   = datasets.flatten()[:, None]
		self.ohe    = preprocessing.OneHotEncoder(sparse=False).fit(self.locs)
		self.n_sets = len(np.unique(self.locs))

	def _transform(self, X, *args, idx=None, zeros=False, **kwargs):
		# if X is not the same shape as locs, and idx=None, we just append zeros
		# otherwise, we append the appropriate one-hot vector corresponding to a 
		# sample's dataset membership
		if ((X.shape[0] == len(self.locs)) or (idx is not None)) and not zeros:
			idx = idx or slice(None)
			loc = self.locs[idx]
			return np.append(X, self.ohe.transform(loc), 1)
		return np.append(X, np.zeros((len(X), self.n_sets)), 1)

	def _inverse_transform(self, X, *args, **kwargs):
		return X[:, :-self.n_sets]