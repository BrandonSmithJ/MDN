'''

'''

from ...utils import get_required, optimize

# Define any optimizable parameters
@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [668]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 1140.25)
	b = kwargs.get('b', 1.91)

	return a * Rrs(668) - b