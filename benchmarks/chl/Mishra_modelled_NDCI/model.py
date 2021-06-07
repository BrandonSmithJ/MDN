'''
Normalized Difference Chlorophyll Index (NDCI) proposed by 
Mishra and Mishra (2012). This version of the model has been 
calibrated using modelled Rrs spectra.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 42.197)
	b = kwargs.get('b', 236.5)
	c = kwargs.get('c', 314.97)

	ratio = (Rrs(708) - Rrs(665)) / (Rrs(708) + Rrs(665))
	return a + b * ratio + c * ratio**2