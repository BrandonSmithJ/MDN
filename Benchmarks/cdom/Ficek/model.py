'''
Ficek a_cdom(440)
'''

from ...utils import get_required, optimize, closest_wavelength

# Define any optimizable parameters
@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = []
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 3.65)
	b = kwargs.get('b', 1.93)
	return a * (Rrs(570) / Rrs(655)) ** -b