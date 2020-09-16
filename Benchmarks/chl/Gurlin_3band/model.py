'''
Three-band ratio algorithm of Gurlin et al. (2011) which 
was calibrated using field measurements of Rrs and Chla
taken from Fremont lakes Nebraska.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 753]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 315.50)
	b = kwargs.get('b', 215.95)
	c = kwargs.get('c', 25.66)
	
	ratio = Rrs(753) * (1/Rrs(665) - 1/Rrs(708))
	return a * ratio**2 + b * ratio + c