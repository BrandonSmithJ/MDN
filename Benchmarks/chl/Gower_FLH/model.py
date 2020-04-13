'''
NASA fluorescence line height (FLH) algo-
rithm presented by Gower et al. (1999). It produces an estimate of the
magnitude of sun induced chlorophyll fluorescence (SICF) at 681 nm
above a baseline interpolated between 665 and 708 nm;
'''

from ..utils import get_required, optimize

@optimize(['a', 'b'], has_default=False)
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 681, 708]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a')
	b = kwargs.get('b')
	return a + b * (Rrs(681) - (Rrs(708) + (Rrs(665) - Rrs(708)) * ((708 - 681)/(708 - 665))))