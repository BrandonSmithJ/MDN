'''
Two-band empirically derived ratio algorithm of Gurlin et al. (2011)
'''

from ...utils import get_required, optimize

@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 25.28)
	b = kwargs.get('b', 14.85)
	c = kwargs.get('c', -15.18)

	ratio = Rrs(708) / Rrs(665)
	return a * ratio**2 + b * ratio + c