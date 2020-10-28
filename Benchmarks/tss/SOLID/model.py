'''
"Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters". S.V. Balasubramanian, et al. (2020).
'''

from ...utils import get_required, optimize, has_band, closest_wavelength, find_wavelength, to_rrs, loadtxt
from ...other.QAA.model import model as QAA
from ....product_estimation import apply_model as mdn_estimate

from scipy.interpolate import CubicSpline as Interpolate
from pathlib import Path
import numpy as np


model_hash = {
	'MOD' : 'cd0c01156295ecdfdb838f27b838c339fb2ee52e135fdd1eebca1fa57cf2e203',
	'MSI' : '684a3cee13d1135778a951ad12e94cfb46a8e1a940d7a70879c4d2ad015ad725',
	'OLCI': 'f5ce0ea3fec891325a3b2ce648607a54fbe7185211108dac138ecd6dcb568e88',
	'OLI' : '1bac89926485e83eaeba287c9de3323ae8f8d294f976b23e6332edb67a23b17e',
	'VI'  : '832abef1b7e6cdca7ce052d380d87123c898e791ea4f9427b5b7b6ab9d1e4943',
}

params = {
	'MSI'   : ([443, 490, 560, 665, 705], 740),
	'OLI'   : ([443, 482, 561, 655], 865),
	'VI'    : ([410, 443, 486, 551, 671], 745),
	'MOD'   : ([412, 443, 488, 555, 667, 678], 748),
	'OLCI'  : ([411, 442, 490, 510, 560, 619, 664, 673, 681], 754),
}


# Define any optimizable parameters
@optimize(['a', 'b', 'c', 'd', 'e', 'f'])
def model(Rrs, wavelengths, sensor, *args, **kwargs):
	sensor = sensor.replace('S2B','MSI').replace('MODA', 'MOD')
	required, upper_band = params[sensor]

	tol = kwargs.get('tol', 10) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 53.736) # Eq. 6
	b = kwargs.get('b', 0.8559) # Eq. 6
	c = kwargs.get('c', 224.43 if sensor == 'OLI' else 207.57) # Eq. 7 & 8
	d = kwargs.get('d', 12.575 if sensor == 'OLI' else 46.78)  # Eq. 7 & 8
	e = kwargs.get('e', 0.5 if sensor == 'OLI' else 1.65)      # a_nap
	f = kwargs.get('f', 0.105) # f/Q

	type1 = (Rrs(485) > Rrs(560)).flatten()
	type2 = (Rrs(560) > Rrs(660)).flatten()
	type3 = (Rrs(660) > Rrs(485)).flatten()

	mdn_kws = { 
		'sensor'    : sensor,
		'product'   : 'bb_p', 
		'use_sim'   : True, 
		'n_iter'    : 10000, 
		'seed'      : 1234,
		'silent'    : True, 
		'model_loc' : Path(__file__).parent.resolve().joinpath('MDN_Model').as_posix(),
		'model_hash': model_hash[sensor],
	}

	estimate = np.empty(type1.shape)
	estimate.fill(np.nan)

	if has_band(upper_band, wavelengths, tol):
		absorb  = Interpolate( *loadtxt('../IOP/aw').T  )
		scatter = Interpolate( *loadtxt('../IOP/bbw').T )
		bbp_NIR = (Rrs(upper_band)*(absorb(upper_band) + e + scatter(upper_band)) - scatter(upper_band)*f) / (f-Rrs(upper_band))
		estimate[type3] = (c * bbp_NIR - d).flatten()[type3]

	ests, idxs  = mdn_estimate(Rrs(required), use_cmdline=False, **mdn_kws)
	bbp_665 = ests[:, idxs['bb_p']][:, find_wavelength(660, required)]
	estimate[type2] = (a * bbp_665 ** b).flatten()[type2]

	bbp = QAA(Rrs(None), wavelengths, sensor, *args, **kwargs)['bbp']
	bbp_665 = get_required(bbp, wavelengths, [])(665)
	estimate[type1] = (a * bbp_665 ** b).flatten()[type1]

	return estimate
