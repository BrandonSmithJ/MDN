from ._CustomTransformer import _CustomTransformer

import numpy as np 


class ExclusionTransformer(_CustomTransformer):
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
		self.transformer = transformer(*transformer_args, **transformer_kwargs)
		self.excl = exclude_slice

	def _fit(self, X, *args, **kwargs):
		cols = np.arange(X.shape[1])
		cols = [c for c in cols if c not in cols[self.excl]]
		self.transformer.fit(X[:, cols])
		self.keep = cols

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