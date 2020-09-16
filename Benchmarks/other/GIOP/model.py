'''
GIOP ocean color reflectance inversion model

P.J. Werdell and 18 co-authors, "Generalized ocean color 
inversion model for retrieving marine inherent optical 
properties," Appl. Opt. 52, 2019-2037 (2013).

GIOP is an ocean reflectance inversion model that
can be configured at run-time.

Requires equally-sized vectors of wavelength and Rrs
plus an estimate of chlorophyll; all other parameterizations
are controlled by the structure 'gopt', described below

Processing comments:
- defaults to GIOP-DC configuration 
- currently requires 412, 443, and 547/555 nm to be present

Outputs are a vector of the magnitudes of the eigenvalues
for adg, bbp, and aph (x), plus modeled spectra of apg,
aph, adg, bbp, and Rrs

Rewritten in python via:
- C implementation (B. Franz, 2008) (https://oceancolor.gsfc.nasa.gov/docs/ocssw/giop_8c_source.html)
- Matlab implementation (J. Werdell, 2013)

Brandon Smith, NASA Goddard Space Flight Center, April 2018
'''

from ...chl.OC.model import model3 as OC3 
from ...utils import get_required, optimize, loadtxt, to_rrs
from ...meta import (
	h0, h1, h2,
	g0_Gordon as g0, 
	g1_Gordon as g1,
)

from scipy.interpolate import CubicSpline as Interpolate
from scipy.optimize import minimize
import numpy as np 


# Define any optimizable parameters
@optimize([])
def model(Rrs, wavelengths, sensor, *args, independent=True, **kwargs):
	''' 
	With independent=False, there is a dependency between sample
	estimations - meaning the estimated parameters can vary wildly
	depending on which samples are passed in.
	'''
	def bricaud(chl, wavelengths):
		data = loadtxt('../IOP/bricaud_1998_aph.txt')
		aphs = (data[:,3] * chl ** (data[:,4]-1)).T
		aphs*= 0.055 / aphs[data[:,0] == 442]
		return Interpolate(data[:,0], aphs)(wavelengths).T

	wavelengths = np.array(wavelengths)
	required = [443, 555]
	tol = kwargs.get('tol', 9) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol)

	aw  = Interpolate( *loadtxt('../IOP/optics_coef.txt', ' ')[:,:2].T )(wavelengths)
	bbw = 0.0038 * (400 / wavelengths) ** 4.32 
	chl = OC3(Rrs(None), wavelengths, sensor).flatten()[:, None]

	rrs = get_required(to_rrs(Rrs(None)), wavelengths, required, tol)
	eta = 2*(1-1.2*np.exp(-0.9*rrs(443)/rrs(555)))
	sdg = 0.018

	aph = bricaud(chl, wavelengths)
	bbp = (443 / wavelengths) ** eta 
	adg = np.exp(-sdg * (wavelengths - 443))
	rrs = rrs(None)

	assert(len(rrs) == len(chl) == len(bbp) == len(aph)), \
		[rrs.shape, chl.shape, bbp.shape, aph.shape]

	if independent:
		aph = aph[:, None]
		bbp = bbp[:, None]
		rrs = rrs[:, None]

	results = []
	for _aph, _bbp, _rrs, _chl in zip(aph, bbp, rrs, chl):

		# Function minimization
		if True:
			def cost_func(guess):
				guess = np.array(guess).reshape((3, -1, 1))
				atot  = aw  + _aph * guess[2] + adg * guess[0]
				bbtot = bbw + _bbp * guess[1]
				u     = bbtot / (atot + bbtot)
				rmod  = g0 * u + g1 * u ** 2
				cost  = np.sum((_rrs-rmod) ** 2, 1) # Sum over bands

				return cost.mean()                  # Average over samples 

			init = [[0.01]*len(_chl), [0.001]*len(_chl), _chl]
			res  = minimize(cost_func, init, tol=1e-6, options={'maxiter':1e3}, method='BFGS')
			# res  = minimize(cost_func, init, tol=1e-6, options={'maxiter':1e3}, method='SLSQP', bounds=[(0, 1e3)]*len(init))
			# res  = minimize(cost_func, init, tol=1e-10, options={'maxiter':1e5}, method='SLSQP')
			x    = np.array(res.x).reshape((3, -1, 1))

		# Linear matrix inversion
		else:
			q = (-g0 + (g0 ** 2 + 4 * g1 * _rrs)**0.5) / (2 * g1)
			b = (bbw * (1-q) - aw * q).T  
			Z = np.vstack([np.atleast_2d(adg) * q, np.atleast_2d(_bbp) * (q-1), np.atleast_2d(_aph) * q]).T
			Q,R = np.linalg.qr(Z)
			x   = np.linalg.lstsq(R, np.linalg.lstsq(R.T, np.dot(Z.T, b))[0])[0]
			r   = b - np.dot(Z, x)
			err = np.linalg.lstsq(R, np.linalg.lstsq(R.T, np.dot(Z.T, r))[0])[0]
			x   = (x + err).flatten().reshape((3, -1, 1))

		madg = x[0] * adg
		mbbp = x[1] * _bbp
		maph = x[2] * _aph
		mchl = x[2]

		mapg = madg + maph
		moda = aw   + mapg
		modb = bbw  + mbbp 
		modx = modb / (modb + moda)
		mrrs = g0 * modx + g1 * modx ** 2

		results.append([mchl, mbbp, madg, maph, mapg, moda, modb])
	return dict(zip(['chl', 'bbp', 'adg', 'aph', 'apg', 'a', 'b'], map(np.vstack, zip(*results))))