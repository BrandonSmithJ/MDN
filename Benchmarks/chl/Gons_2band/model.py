'''
Semi-analytical algorithm presented by Gons et al. (2002, 2005, 2008) 
which incorporates information on water absorption and backscattering
in relation to the MERIS red-NIR reflectance ratio
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 778]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 1.063)
	b = kwargs.get('b', 0.016)
	
	pi = 355 / 113. # pi approximation
	bb = 1.61 * pi * Rrs(708) / (0.082 - 0.6 * pi * Rrs(778))
	return ((Rrs(708) / Rrs(665)) * (0.7 + bb) - 0.4 - bb**a) / b 