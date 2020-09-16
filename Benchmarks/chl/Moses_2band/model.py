'''
Two-band ratio algorithm of Dall'Olmo et al.
(2003), Moses et al. (2009) and Gitelson et al. (2011), originally proposed
by Gitelson and Kondratyev (1991) and later adapted to MERIS
bands. This is an empirical formula based on a linear relationship between
in-situ Chla and the ratio of MERIS satellite remote sensing reflectance,
measured at NIR, Rrs(708), and red, Rrs(665), wavelengths.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 61.324)
	b = kwargs.get('b', 37.94)
	return a * (Rrs(708) / Rrs(665)) - b 
