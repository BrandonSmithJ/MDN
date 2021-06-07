from ._CustomTransformer import _CustomTransformer

import numpy as np 


class LogTransformer(_CustomTransformer):
	''' Transform into log domain '''
	def _transform(self, X, *args, **kwargs):         return np.log(X)
	def _inverse_transform(self, X, *args, **kwargs): return np.exp(X)

