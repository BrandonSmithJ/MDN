from ._CustomTransformer import _CustomTransformer


class IdentityTransformer(_CustomTransformer):
	''' No transformation '''
	def _transform(self, X, *args, **kwargs):         return X
	def _inverse_transform(self, X, *args, **kwargs): return X

