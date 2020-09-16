'''
Band index algorithm presented by Yang et al.
(2010), which is based on a conceptual model (Gitelson et al., 2008)
that adopts relevant wavebands according to their sensitivity to water
absorption properties
it is assumed R rs (665) has maximum sensitivity to phytoplankton
absorption, R rs (708) is insensitive to phytoplankton absorption but
comparably sensitive to CDOM and R rs (753) is insensitive to phyto-
plankton and CDOM absorption and is mainly influenced by back-
scattering. Chla is estimated from a three-band index using a simple
empirical formula
coefficients a = 161.24 and b = 28.04 have been calibrated for
lakes in Japan and China
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 753]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 161.24)
	b = kwargs.get('b', 28.04)

	index = ((1/Rrs(665) - 1/Rrs(708)) / (1/Rrs(753) - 1/Rrs(708)))
	return a * index + b