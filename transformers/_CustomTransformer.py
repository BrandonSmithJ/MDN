from sklearn.base import TransformerMixin


class _CustomTransformer(TransformerMixin):
	''' Data transformer class which validates data shapes. 
		Child classes should override _fit, _transform, _inverse_transform '''
	_input_shape  = None 
	_output_shape = None

	def fit(self, X, *args, **kwargs):				 
		self._input_shape = X.shape[1]
		self._fit(X.copy(), *args, **kwargs)
		return self 

	def transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._input_shape)
		X = self._transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._output_shape)
		self._output_shape = X.shape[1]
		return X 

	def inverse_transform(self, X, *args, **kwargs):
		self._validate_shape(X, self._output_shape)
		X = self._inverse_transform(X.copy(), *args, **kwargs)
		self._validate_shape(X, self._input_shape)
		self._input_shape = X.shape[1]
		return X 

	@staticmethod
	def config_info(*args, **kwargs):                 return '' # Return any additional info to construct model config
	def _fit(self, X, *args, **kwargs):               pass
	def _transform(self, X, *args, **kwargs):         raise NotImplemented
	def _inverse_transform(self, X, *args, **kwargs): raise NotImplemented
	def _validate_shape(self, X, shape):              assert(shape is None or X.shape[1] == shape), \
		f'Number of data features changed: expected {shape}, found {X.shape[1]}'