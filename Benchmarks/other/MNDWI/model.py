'''

'''

from ...utils import get_required, optimize

# Define any optimizable parameters
@optimize([])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [560, 1600]
	tol = kwargs.get('tol', 20) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	return (Rrs(560) - Rrs(1600)) / (Rrs(560) + Rrs(1600))