from ._CustomTransformer import _CustomTransformer

import numpy as np 


class TanhTransformer(_CustomTransformer):
	''' tanh-estimator (Hampel et al. 1986; Latha & Thangasamy, 2011) '''
	scale = 0.01

	def _fit(self, X, *args, **kwargs):
		m = np.median(X, 0)
		d = np.abs(X - m)

		a = np.percentile(d, 70, 0)
		b = np.percentile(d, 85, 0)
		c = np.percentile(d, 95, 0)

		Xab = np.abs(X)
		Xsi = np.sign(X)
		phi = np.zeros(X.shape)
		idx = np.logical_and(0 <= Xab, Xab < a)
		phi[idx] = X[idx]
		idx = np.logical_and(a <= Xab, Xab < b)
		phi[idx] = (a * Xsi)[idx]
		idx = np.logical_and(b <= Xab, Xab < c)
		phi[idx] = (a * Xsi * ((c - Xab) / (c - b)))[idx]

		self.mu_gh  = np.mean(phi, 0)
		self.sig_gh = np.std(phi, 0) 

	def _transform(self, X, *args, **kwargs):
		return 0.5 * (np.tanh(self.scale * ((X - self.mu_gh)/self.sig_gh)) + 1)

	def _inverse_transform(self, X, *args, **kwargs):
		return ((np.tan(X * 2 - 1) / self.scale) * self.sig_gh) + self.mu_gh
	
