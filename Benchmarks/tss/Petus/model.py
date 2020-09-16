'''

'''

from ...utils import get_required, optimize

# Define any optimizable parameters
@optimize(['a', 'b', 'c'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [668]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 12450)
	b = kwargs.get('b', 666.1)
	c = kwargs.get('c', 0.48)

	return a * Rrs(668) ** 2 + b * Rrs(668) + c