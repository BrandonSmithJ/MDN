'''
Simplified version of Gilerson et al. (2010) which
relates Chla to the NIR-red reflectance band ratio through a simple
power function.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'], has_default=False)
def model(Rrs, wavelengths, *args, **kwargs):
	required = []
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a')
	b = kwargs.get('b')
	return a * (Rrs(708) / Rrs(665)) ** b