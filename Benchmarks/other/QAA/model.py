'''
QAA v6 
http://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf 
http://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm

There are inconsistencies between the pdf definition, and the spreadsheet
implementation. Notably, the spreadsheet uses Rrs throughout rather than
rrs. As well, absorption / scatter values are slightly off, and the a_ph
calculation uses wavelength-specific absorption rather than solely the 443
band.

Here, we use the pdf definition in all cases except the a_ph calculation -
using wavelength absorption prevents higher band a_ph estimates from 
flattening (bands > ~500nm). Where exact bands are requested (e.g. the 
reference band lambda0), this implementation uses the nearest available
band. This impacts the exact absorption/scattering values used, as well as
the calculation of xi with the band difference in the exponent. 

551 is also used to find the closest 555nm band, in order to avoid using 
the 555nm band of MODIS (which is a land-focused band). 
'''

from ...utils import get_required, optimize, loadtxt, to_rrs, closest_wavelength
from ...meta import (
	h0, h1, h2,
	g0_QAA as g0, 
	g1_QAA as g1,
)

from scipy.interpolate import CubicSpline as Interpolate
from pathlib import Path
import numpy as np


# Define any optimizable parameters
@optimize([])
def model(Rrs, wavelengths, *args, lambda_reference=None, **kwargs):
	wavelengths = np.array(wavelengths)
	required = [443, 490, 550, 670]
	tol = kwargs.get('tol', 9) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)
	rrs = get_required(to_rrs(Rrs(None)), wavelengths, required, tol)

	absorb  = Interpolate( *loadtxt('../IOP/aw').T  )
	scatter = Interpolate( *loadtxt('../IOP/bbw').T )

	get_band   = lambda k: closest_wavelength(k, wavelengths, tol=tol)
	functional = lambda v: get_required(v, wavelengths)

	# Invert rrs formula to find u
	u = functional( (-g0 + (g0**2 + 4 * g1 * rrs(None)) ** 0.5) / (2 * g1) )

	# Next couple steps depends on if Rrs(670) is lt/gt 0.0015
	QAA_v5 = Rrs(670) < 0.0015
	a_full = np.zeros(QAA_v5.shape) # a(lambda_0)
	b_full = np.zeros(QAA_v5.shape) # b_bp(lambda_0)
	l_full = np.zeros(QAA_v5.shape) # lambda_0

	# --------------------
	# If Rrs(670) < 0.0015 (QAA v5)
	if QAA_v5.sum():
		lambda0 = get_band(551)
		a_w = absorb(lambda0)
		b_w = scatter(lambda0)
		chi = np.log10( (rrs(443) + rrs(490)) / 
						(rrs(lambda0) + 5 * (rrs(670) / rrs(490)) * rrs(670)) )

		a = a_w + 10 ** (h0 + h1 * chi + h2 * chi**2)
		b = (u(lambda0) * a) / (1 - u(lambda0)) - b_w

		a_full[QAA_v5] = a[QAA_v5]
		b_full[QAA_v5] = b[QAA_v5]
		l_full[QAA_v5] = lambda0
	# --------------------

	# --------------------
	# else (QAA v6)
	if (~QAA_v5).sum():
		lambda0 = get_band(670)
		a_w = absorb(lambda0)
		b_w = scatter(lambda0)

		a = a_w + 0.39 * ( Rrs(670) / (Rrs(443) + Rrs(490)) ) ** 1.14
		b = (u(lambda0) * a) / (1 - u(lambda0)) - b_w

		a_full[~QAA_v5] = a[~QAA_v5]
		b_full[~QAA_v5] = b[~QAA_v5]
		l_full[~QAA_v5] = lambda0
	# --------------------

	# Back to the same steps for all data
	a0 = a_full
	b0 = b_full
	l0 = l_full

	eta = 2 * (1 - 1.2 * np.exp(-0.9 * rrs(443) / rrs(551)))

	b = b0 * (l0 / wavelengths) ** eta
	a = (1 - u(None)) * (scatter(wavelengths) + b) / u(None)
	a = functional(a)

	# Now decompose the absorption
	zeta = 0.74 + (0.2 / (0.8 + rrs(443) / rrs(551)))
	S    = 0.015 + (0.002 / (0.6 + rrs(443) / rrs(551)))
	xi   = np.exp(S * (get_band(443)-get_band(412))) 

	# {a_g443, a_dg, a_ph} all require a 412nm band (thus are not available for e.g. OLI)
	a_g443 =  (a(412) - zeta * a(443)) / (xi - zeta) \
			- (absorb(get_band(412)) - zeta * absorb(get_band(443))) / (xi - zeta)

	a_dg = a_g443 * np.exp(S * (get_band(443) - wavelengths))
	a_ph = a(None) - a_dg - absorb(wavelengths) # differs from pdf doc; shown in spreadsheet	

	# Remove negatives
	b[b < 0] = 1e-5
	#a_ph[a_ph < 0] = 1e-5

	# QAA-CDOM - Zhu & Yu 2013
	a_p = 0.63 * b ** 0.88
	a_g = a(None) - absorb(get_band(443)) - a_p  
	
	if lambda_reference is not None:
		return  functional(b)(lambda_reference),    \
				functional(a_ph)(lambda_reference), \
				functional(a_g)(lambda_reference), eta, S

	# Return all backscattering and absorption parameters
	return {
		'a'  : a(None),
		'aph': a_ph,
		'ap' : a_p, # a_ph + (a_dg - a_g)
		'ag' : a_g,
		'apg': a_ph + a_dg,
		'adg': a_dg, 
		'b'  : b + b_w,
		'bbp': b,
	}