'''
Fluorescence Line Height (FLH) (Letelier and Abbott, 1996; Gower et al., 1999). 
FLH produces an estimate of the magnitude of sun induced chlorophyll 
fluorescence (SICF) at 681nm above a baseline interpolated between 665 and 708nm. 

Two "outer" bands are meant to estimate the baseline or elastic
reflectance (mostly outside the fluorescence spectrum), while the 
middle band determines the height.
"Bio-optical Modeling and Remote Sensing of Inland Waters" Pg 211

FLH = L_2 - k*L_1 - (1-k)*L_3
or
FLH = L_2 - (L_3 + k*(L_1 - L_3))

k = (lambda_3 - lambda_2) / (lambda_3 - lambda_1)
'''

from ...utils import get_required, optimize

params = {
	'MOD'   : [665, 680, 750],
	'MODA'  : [665, 680, 750],
	'MODT'  : [665, 680, 750],

	'MERIS' : [665, 680, 709],

	'MSI'   : [665, 705, 740],
	'S2A'   : [665, 705, 740],
	'S2B'   : [665, 705, 740],

	'OLCI'  : [665, 680, 750],
	'S3A'   : [665, 680, 750],
	'S3B'   : [665, 680, 750],
}

@optimize(['a', 'b'])
def model(Rrs, wavelengths, sensor, *args, **kwargs):
	assert(sensor in params), f'No wavelengths defined for FLH using {sensor}'
	required = L1, L2, L3 = params[sensor]

	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 0)
	b = kwargs.get('b', 1)
	k = (L3 - L2) / (L3 - L1)
	return a + b * (Rrs(L2) - k*Rrs(L1) - (1-k)*Rrs(L3))