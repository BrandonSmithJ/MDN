from tensorflow.python.ops import math_ops
import tensorflow as tf 


class MSA(tf.keras.metrics.Mean):
	''' Mean Symmetric Accuracy '''

	def __init__(self, extract_estimate=lambda x:x, invert_scaling=lambda x:x):
		super(MSA, self).__init__(name='MSA', dtype=float)
		self._extract = extract_estimate
		self._invert  = invert_scaling

	def update_state(self, y_true, y_pred, *args, **kwargs):
		y_true = self._invert(y_true)
		y_pred = self._invert( self._extract(y_pred) )
		value  = tf.math.abs( tf.math.log(y_pred / y_true) )
		return super().update_state(value, *args, **kwargs) 
 
	def result(self):
		return 100. * (tf.math.exp(math_ops.div_no_nan(self.total, self.count)) - 1.)
