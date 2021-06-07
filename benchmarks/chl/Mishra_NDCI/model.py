'''
Normalized Difference Chlorophyll Index (NDCI) proposed by 
Mishra and Mishra (2012) calibrated using field data collected 
from Chesapeake and Delaware Bay.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 14.039)
	b = kwargs.get('b', 86.115)
	c = kwargs.get('c', 194.325)

	ratio = (Rrs(708) - Rrs(665)) / (Rrs(708) + Rrs(665))
	return a + b * ratio + c * ratio**2