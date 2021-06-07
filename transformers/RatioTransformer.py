from ._CustomTransformer import _CustomTransformer

from itertools import combinations
import numpy as np 


class RatioTransformer(_CustomTransformer):	
	''' Add ratio features '''

	def __init__(self, wavelengths, *args, excl_Rrs=False, all_ratio=False, **kwargs):
		assert(not all_ratio or len(wavelengths) < 16), 'Too many features would be created with > 15 bands (i.e. > 1000 features)'
		self.wavelengths = list(wavelengths)
		self.all_ratio   = all_ratio
		self.excl_Rrs    = excl_Rrs
		self.labels      = []

	@staticmethod
	def config_info(*args, **kwargs):
		transformer = RatioTransformer(*args, **kwargs)
		return f'{transformer.get_n_features()} features: {transformer.labels}'

	def get_n_features(self):
		random_data = np.ones((2, len(self.wavelengths)))
		return self._transform(random_data).shape[-1]

	def _fit(self, X, *args, **kwargs):
		self.shape = X.shape[1]

	def _transform(self, X, *args, **kwargs):		 
		''' Ratios based on literature '''
		from ..Benchmarks.utils import get_required, has_band, closest_wavelength
		self.labels = []

		x     = np.atleast_2d(X)
		x_new = []
		Rrs   = get_required(x, self.wavelengths, tol=11)

		def BR2(L1, L2):
			self.labels.append(f'{L1}/{L2}')
			return [Rrs(L1) / Rrs(L2)]

		def BR3(L1, L2, L3):
			self.labels.append(f'{L2}/{L1}-{L2}/{L3}')
			return [Rrs(L2)/Rrs(L1) - Rrs(L2)/Rrs(L3)]

		def LH(L1, L2, L3):
			self.labels.append(f'{L1}-{L2}-{L3}')
			c = (L3 - L2) / (L3 - L1)
			return [Rrs(L2) - c*Rrs(L1) - (1-c)*Rrs(L3)]

		def PCA(n_components=1):
			from sklearn.decomposition import PCA as PCA_sklearn
			values = PCA_sklearn(n_components).fit_transform( Rrs(None) )
			self.labels += [f'PCA{i}' for i in range(values.shape[-1])]
			return [values[:, i] for i in range(values.shape[-1])]

		features = (
			[(BR2, wvl) for wvl in combinations(self.wavelengths, 2)] + 
			[(BR3, wvl) for wvl in combinations(self.wavelengths, 3)] + 
			[(LH,  wvl) for wvl in combinations(self.wavelengths, 3)]  
		) if self.all_ratio else [
			(BR2, (745, 783)), 
			(BR2, (665, 705)),
			(BR2, (560, 705)),
			(BR2, (560, 665)),
			(BR2, (490, 560)),
			(BR2, (443, 490)),
			(BR2, (665, 620)),
			(BR2, (620, 560)),

			(LH, (490, 560, 665)),
			(LH, (665, 705, 745)),
			(LH, (705, 745, 783)),
			(LH, (560, 620, 665)),
		]

		for function, wvls in features:
			if all(has_band(wvl, self.wavelengths, tol=11) for wvl in wvls):
				x_new += function(*closest_wavelength(wvls, self.wavelengths, tol=11))

		# x_new.append((R705 - R665) / (R705 + R665)); self.labels.append('NDCI') # NDCI
		# x_new.append((R560 - R1613) / (R560 + R1613)); self.labels.append('MNDWI') # NDWI
		# x_new.append((1/R665 - 1/R705) / (1/R783 - 1/R705)); self.labels.append('YangBI') # Yang band index

		for xv in x_new: assert(np.all(np.isfinite(xv))), X[~np.isfinite(xv)]
		assert(len(x_new) == len(self.labels)), f'Mismatch between features and labels: {len(x_new)} vs {len(self.labels)}'
		if self.excl_Rrs:
			return np.hstack([v.flatten()[:, None] for v in x_new])
		return np.append(x, np.hstack([v.flatten()[:, None] for v in x_new]), -1)


	def _inverse_transform(self, X, *args, **kwargs): 
		return np.array(X)[:, :self.shape]
