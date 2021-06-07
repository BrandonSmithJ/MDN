from ._CustomTransformer import _CustomTransformer


class ColumnSelectionTransformer(_CustomTransformer):
	''' Reduce columns to specified selections (feature selection) '''
	def __init__(self, columns, *args, **kwargs): self._c = columns 
	def _transform(self, X, *args, **kwargs):     return X[:, self._c]
