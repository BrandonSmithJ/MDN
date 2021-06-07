'''
The three-band algorithm developed by Moses et al. (2009) for MERIS,
and adapted by Gitelson et al. (2011) to include Rrs measured at 753 nm;
In theory, the combination of three bands alters the model sensitivity to
the presence of optically active constituents by removing the effects of
SPM and CDOM (R rs (665) and R rs (708) are comparably influenced by
SPM and CDOM and R rs (753) is mainly driven by backscattering).
'''

from ...utils import get_required, optimize

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 753]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 232.329)
	b = kwargs.get('b', 23.174)
	return a * (Rrs(753) * (1/Rrs(665) - 1/Rrs(708))) + b
