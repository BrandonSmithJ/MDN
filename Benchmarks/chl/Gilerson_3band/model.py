'''
advanced three-band semi-analytical algo-
rithm proposed by Gilerson et al. (2010). As per Model E, the three-
band model is based on a semi-analytical expression for the red-NIR
ratio of reflectance in combination with water absorption
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 753]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 0.022)
	b = kwargs.get('b', 0.8897)
	return ((2.494 / a) * (Rrs(753) * (1/Rrs(665) - 1/Rrs(708))) + ((0.7864 / a) - (0.4245 / a))) ** (1/b)
