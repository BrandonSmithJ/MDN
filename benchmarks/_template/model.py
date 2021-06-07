'''
Copy this folder into a product directory, and rename it to be the algorithm name
'''

from ...utils import get_required, optimize

# Define any optimizable parameters
@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = []
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', )
	b = kwargs.get('b', )
	return