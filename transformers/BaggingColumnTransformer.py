from ._CustomTransformer import _CustomTransformer

import numpy as np 


class BaggingColumnTransformer(_CustomTransformer):
	''' 
	Randomly select a percentage of columns to drop, always keeping the
	band features (first n_bands columns). Optionally, the last n_extra 
	columns will also be always kept as additional features.
	'''

	def __init__(self, n_bands, *args, n_extra=0, percent=0.75, seed=None, **kwargs):
		self.n_bands = n_bands
		self.n_extra = n_extra
		self.percent = percent
		self.random  = np.random.RandomState(seed)

	def _fit(self, X, *args, **kwargs):
		shp  = X.shape[1] - self.n_bands
		ncol = int(shp*self.percent)
		cols = np.arange(shp-self.n_extra) + self.n_bands
		self.random.shuffle(cols)
		new_cols  = list(cols[:ncol])
		if self.n_extra: new_cols += list(X.shape[1]-(np.arange(self.n_extra)+1))
		self.cols = np.append(np.arange(self.n_bands), new_cols, 0)
		# print(f'Reducing bands from {shp} ({X.shape[1]} total) to {ncol} ({len(self.cols)} total) ({self.cols})')

	def _transform(self, X, *args, **kwargs):
		return X[:, self.cols.astype(int)]