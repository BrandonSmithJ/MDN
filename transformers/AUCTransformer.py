from ._CustomTransformer import _CustomTransformer

import numpy as np 


class AUCTransformer(_CustomTransformer):
	''' Area under the curve normalization '''

	def __init__(self, wavelengths, *args, **kwargs): 
		self.wavelengths = wavelengths
	
	def _transform(self, X, *args, **kwargs):     
		area = np.trapz(X, self.wavelengths, axis=1)[:, None]
		return X/area#np.append(X / area, area, 1) 
