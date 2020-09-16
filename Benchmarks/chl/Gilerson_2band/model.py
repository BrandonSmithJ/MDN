'''
advanced two-band semi-analytical algorithm
proposed by Gilerson et al. (2010). While this is governed by the ratio
of NIR to red reflectance, model coefficients are determined analytically
from individual absorption components contributing to the total IOPs of
the water body. It is assumed that the water term dominates (at red –
NIR wavelengths) where Chla concentration is > 5 mg m −3 , and that
the contribution to absorption by CDOM and backscattering terms are
significantly smaller.
a may be determined empirically and b is parameterised to fit
the data. The water term becomes less dominant when
Chla < 5 mg m −3 , and therefore the assumed negligibility of the in-
fluence of CDOM and SPM is no longer valid under these conditions.
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 0.022)
	b = kwargs.get('b', 0.8897)
	return ((0.7864 / a) * (Rrs(708) / Rrs(665)) - (0.4245 / a)) ** (1/b)
