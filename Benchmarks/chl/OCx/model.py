from ...utils import get_required, optimize
from ..OC.model import OC
import numpy as np


@optimize(['a', 'b'])
def model(Rrs, wavelengths, sensor, *args, OCI=False, **kwargs):
	req = [443, 555, 670]
	tol = kwargs.get('tol', 15)
	Rrs = get_required(Rrs, wavelengths, req, tol)

	if OCI:
		lo  = kwargs.get('a', 0.25)
		hi  = kwargs.get('b', 0.30)
	else:
		lo  = kwargs.get('a', 0.15)
		hi  = kwargs.get('b', 0.20)

	CI  = Rrs(551) - (Rrs(443) + (555 - 443) / (670 - 443) * (Rrs(670) - Rrs(443)))
	CI[CI > 0] = 0
	CI_Chl = 10 ** (-0.4909 + 191.6590*CI)
	CI2    = CI_Chl.copy()
	CI2[CI2 <  lo] = lo
	CI2[CI2 >= hi] = hi

	oc  = OC(Rrs(None), wavelengths, sensor, num=3)
	w_1 = (CI2 - lo) / (hi - lo)
	w_2 = np.abs(w_1 - 1)
	return np.nansum([w_1 * oc, w_2 * CI_Chl], 0)
