from ...utils import get_required, optimize
from ..Gilerson_2band import model as two_band
from ..OCx import model as OCx
import numpy as np


@optimize(['a', 'b'])
def model(Rrs, wavelengths, sensor, *args, **kwargs):
	req = [665, 708]
	tol = kwargs.get('tol', 15)
	Rrs = get_required(Rrs, wavelengths, req, tol)

	lo  = kwargs.get('a', 0.75)
	hi  = kwargs.get('b', 1.15)

	oci = OCx.model(Rrs(None), wavelengths, sensor, OCI=True)
	g2b = two_band.model(Rrs(None), wavelengths)
	phi = Rrs(708) / Rrs(665)
	phi[phi < lo] = lo
	phi[phi > hi] = hi 

	w_1 = (phi - lo) / (hi - lo)
	w_2 = np.abs(w_1 - 1)
	return np.nansum([w_1 * g2b, w_2 * oci], 0) 