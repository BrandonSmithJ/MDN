'''

'''

from ...utils import get_required, optimize, has_band
import numpy as np 

# Define any optimizable parameters
@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [482, 561, 665]
	tol = kwargs.get('tol', 15) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 531.5)
	b = kwargs.get('b', 37150)
	c = kwargs.get('c', 1751)

	type1 = (Rrs(482) > Rrs(561)).flatten()
	type2 = (Rrs(561) > Rrs(665)).flatten()
	type3 = (Rrs(665) > Rrs(482)).flatten()

	estimate = np.empty(type1.shape)
	estimate.fill(np.nan)

	if has_band(865, wavelengths, tol):
		estimate[type3] = ((Rrs(865) * np.pi * b) + (Rrs(865) * np.pi * c)).flatten()[type3]
	estimate[~type3] = (Rrs(665) * np.pi * a).flatten()[~type3]
	return estimate