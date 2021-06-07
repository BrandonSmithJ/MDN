'''
Mannino a_cdom(443)
'''

from ...utils import get_required, optimize
import numpy as np 

# Define any optimizable parameters
@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [490, 555]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 0.0736)
	b = kwargs.get('b', 0.408)
	c = kwargs.get('c', 0.173)
	return -a * np.log(b * Rrs(490) / Rrs(555) - c)