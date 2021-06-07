'''
"Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters". S.V. Balasubramanian, et al. (2020).
'''

from ...utils import get_required, optimize, has_band, closest_wavelength, find_wavelength, to_rrs, loadtxt
from ...multiple.QAA.model import model as QAA
from ....product_estimation import apply_model as mdn_estimate

from scipy.interpolate import CubicSpline as Interpolate
from pathlib import Path
import numpy as np



sensor_parameters = {
	'MOD-SOLID'  : {
		'wavelengths' : [412, 443, 488, 555, 667, 678],
		'upper_band'  : 748,
		'model_uid'   : 'f7581d07348bd78e726d69bab1e524bf6b0771a5cd3c79a13640c51fb477f0af',
	},

	'MSI-SOLID'  : {
		'wavelengths' : [443, 490, 560, 665, 705],
		'upper_band'  : 740,
		'model_uid'   : '33b1543486da13f05b0373a0a4d7d4913be43ad86b8376060de40cae45f440b2',
	},

	'OLCI-SOLID' : {
		'wavelengths' : [411, 442, 490, 510, 560, 619, 664, 673, 681],
		'upper_band'  : 754,
		'model_uid'   : '13d020594e3a5bb2ac1b337f5e7a5f8f60d10c00bed68be7251bdedfd6c7f0b3',
	},

	'OLI-SOLID'  : {
		'wavelengths' : [443, 482, 561, 655],
		'upper_band'  : 865,
		'model_uid'   : '0ec801772974095081a72534d306c1a715272008c7bbadfff4c722663d2bdf9f',
	},

	'VI-SOLID'   : {
		'wavelengths' : [410, 443, 486, 551, 671],
		'upper_band'  : 745,
		'model_uid'   : '515dd6c26349b10e489c1858a70ccdc1c087efa07e898a3eccc237ef8f29b9fe',
	},
}

sensor_name_replacements = {
	'MODA' : 'MOD-SOLID',
	'MODT' : 'MOD-SOLID',
	'MOD'  : 'MOD-SOLID',

	'S2A'  : 'MSI-SOLID',
	'S2B'  : 'MSI-SOLID',
	'MSI'  : 'MSI-SOLID',

	'S3A'  : 'OLCI-SOLID',
	'S3B'  : 'OLCI-SOLID',
	'OLCI' : 'OLCI-SOLID',

	'OLI'  : 'OLI-SOLID',

	'VI'   : 'VI-SOLID',
	'VIIRS': 'VI-SOLID',
}


# Define any optimizable parameters
@optimize(['a', 'b', 'c', 'd', 'e', 'f'])
def model(Rrs, wavelengths, sensor, *args, **kwargs):
	sensor = sensor_name_replacements.get(sensor, sensor)
	params = sensor_parameters[sensor]

	required   = params['wavelengths']
	upper_band = params['upper_band']
	model_uid  = params['model_uid']

	tol = kwargs.get('tol', 11) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 53.736) # Eq. 6
	b = kwargs.get('b', 0.8559) # Eq. 6
	c = kwargs.get('c', 224.43 if sensor == 'OLI-SOLID' else 207.57) # Eq. 7 & 8
	d = kwargs.get('d', 12.575 if sensor == 'OLI-SOLID' else 46.78)  # Eq. 7 & 8
	e = kwargs.get('e', 0.5    if sensor == 'OLI-SOLID' else 1.65)   # a_nap
	f = kwargs.get('f', 0.105) # f/Q

	mdn_kws = { 
		'sensor'    : sensor,
		'seed'      : 42,
		'use_sim'   : True, 
		'silent'    : not kwargs.get('verbose', False), 
		'model_loc' : Path(__file__).parent.resolve().joinpath('MDN_weights').as_posix(),
		'model_uid' : model_uid,
	}

	# Type 2 water estimates
	estimates, slices = mdn_estimate(Rrs(required), use_cmdline=False, **mdn_kws)
	bbp_MDN  = estimates[:, slices['bb_p']]
	bbp_665  = get_required(bbp_MDN, required, [], tol)(665)
	estimate = (a * bbp_665 ** b).flatten()
	type2    = ((Rrs(660) < Rrs(560)) & (Rrs(660) > Rrs(490))).flatten()

	# Type 3 water estimates
	if has_band(upper_band, wavelengths, tol):
		absorb  = Interpolate( *loadtxt('../IOP/aw').T  )(upper_band)
		scatter = Interpolate( *loadtxt('../IOP/bbw').T )(upper_band)
		Rrs_NIR = Rrs(upper_band)
		bbp_NIR = (Rrs_NIR * (absorb + e + scatter) - scatter * f) / (f - Rrs_NIR)
		type3   = ((Rrs(660) > Rrs(560)) & (Rrs_NIR > 0.01)).flatten()
		estimate[type3] = (c * bbp_NIR - d).flatten()[type3]
	else: type3 = False 

	# Type 1 water estimates
	type1   = (Rrs(560) < Rrs(490)).flatten() & ~type2 & ~type3 
	bbp_QAA = QAA(Rrs(None), wavelengths, sensor, *args, **kwargs)['bbp']
	bbp_665 = get_required(bbp_QAA, wavelengths, [], tol)(665)
	estimate[type1] = (a * bbp_665 ** b).flatten()[type1]

	return estimate
