'''

'''

from ...utils import get_required, optimize
import numpy as np 

# Define any optimizable parameters
@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 355.85)
	b = kwargs.get('b', 1.74)
	c = kwargs.get('c', 1728)

	return b + (a * np.pi * Rrs(665)) / (1 - (np.pi * Rrs(665)) / c)