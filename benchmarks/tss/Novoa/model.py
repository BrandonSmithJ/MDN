'''
Novoa et al. 2017 - Gironde Model (Table 3)
'''

from ...utils import get_required, optimize, has_band
import numpy as np 

# Define any optimizable parameters
@optimize(['a', 'b', 'c', 'd'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [482, 561, 665]
	tol = kwargs.get('tol', 15) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', 130.1)
	b = kwargs.get('b', 531.5)
	c = kwargs.get('c', 37150)
	d = kwargs.get('d', 1751)

	ga = kwargs.get('ga', 0.007) # S_GL95-
	gb = kwargs.get('gb', 0.016) # S_GL95+
	ha = kwargs.get('ha', 0.080) # S_GH95-
	hb = kwargs.get('hb', 0.120) # S_GH95+

	p561 = np.pi * Rrs(561).flatten()
	p655 = np.pi * Rrs(665).flatten()
	p865 = np.pi * Rrs(865, validate=False).flatten()

	linear_green = a * p561
	linear_red   = b * p655
	poly_nir     =(c * p865 + d) * p865

	g_alpha = np.log(gb / p655) / np.log(gb / ga)
	g_beta  = np.log(p655 / ga) / np.log(gb / ga)
	h_alpha = np.log(hb / p655) / np.log(hb / ha)
	h_beta  = np.log(p655 / ha) / np.log(hb / ha)

	type_a  = p655 < ga
	type_ab = (ga <= p655) & (p655 <= gb)
	type_b  = (gb <  p655) & (p655 <  ha)
	type_bc = (ha <= p655) & (p655 <= hb)
	type_c  = hb < p655 

	estimate = np.empty(type_a.shape)
	estimate.fill(np.nan)

	estimate[type_a]  = linear_green[type_a]
	estimate[type_ab] = (g_alpha * linear_green + g_beta * linear_red)[type_ab]
	estimate[type_b]  = linear_red[type_b]

	if has_band(865, wavelengths, tol):
		estimate[type_bc] = (h_alpha * linear_red + h_beta * poly_nir)[type_bc]
		estimate[type_c]  = poly_nir[type_c]
	return estimate