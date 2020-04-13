from scipy.interpolate import Akima1DInterpolator as Interpolate
from pathlib import Path
import numpy as np

# SeaWiFS coefficients
h0 = -1.14590292783408 
h1 = -1.36582826429176
h2 = -0.469266027944581 

# Gordon
g0 = 0.0949
g1 = 0.0794

# Lee
g0 = 0.084
g1 = 0.17

# QAA
g0 = 0.08945
g1 = 0.1247  


def water_interpolators():
	root_dir = Path(__file__).parent
	water_absorption_file = root_dir.joinpath('IOP', 'aw')
	water_scattering_file = root_dir.joinpath('IOP', 'bbw')

	a_data = np.loadtxt(water_absorption_file, delimiter=',')
	s_data = np.loadtxt(water_scattering_file, delimiter=',')
	return Interpolate(*a_data.T), Interpolate(*s_data.T)


find = lambda k, wavelength: np.abs(wavelength - k).argmin() 	# Index of closest wavelength
key  = lambda k, wavelength: wavelength[find(k, wavelength)]	# Value of closest wavelength

to_rrs = lambda Rrs: Rrs / (0.52 + 1.7 * Rrs)
to_Rrs = lambda rrs: (rrs * 0.52) / (1 - rrs * 1.7)


def QAA(data, wavelength, lambda_reference=None):
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
	the 555nm band of MODIS (which is a land band). 
	'''
	absorb, scatter = water_interpolators()
	wavelength = np.array(wavelength)

	# Functional interface into matrix, returning values in column with closest wavelength
	idx = lambda f: (lambda k: f[:, find(k, wavelength)][:, None] if k is not None else f)
	Rrs = idx( np.atleast_2d(data) )
	rrs = idx( to_rrs(Rrs(None)) )

	# Invert rrs formula to find u
	u  = idx( (-g0 + (g0**2 + 4 * g1 * rrs(None)) ** 0.5) / (2 * g1) )

	# Next couple steps depends on if Rrs(670) is lt/gt 0.0015
	QAA_v5 = Rrs(670) < 0.0015
	a_full = np.zeros(QAA_v5.shape) # a(lambda_0)
	b_full = np.zeros(QAA_v5.shape) # b_bp(lambda_0)
	l_full = np.zeros(QAA_v5.shape) # lambda_0

	# --------------------
	# If Rrs(670) < 0.0015 (QAA v5)
	if QAA_v5.sum():
		lambda0 = key(551, wavelength)
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
		lambda0 = key(670, wavelength)
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

	b = b0 * (l0 / wavelength) ** eta
	a = (1 - u(None)) * (scatter(wavelength) + b) / u(None)
	a = idx(a)

	# Now decompose the absorption
	zeta = 0.74 + (0.2 / (0.8 + rrs(443) / rrs(551)))
	S    = 0.015 + (0.002 / (0.6 + rrs(443) / rrs(551)))
	xi   = np.exp(S * (key(443, wavelength)-key(412, wavelength))) 

	# {a_g443, a_dg, a_ph} all require a 412nm band (thus are not available for e.g. OLI)
	a_g443 =  (a(412) - zeta * a(443)) / (xi - zeta) \
			- (absorb(key(412, wavelength)) - zeta * absorb(key(443, wavelength))) / (xi - zeta)

	a_dg = a_g443 * np.exp(S * (key(443, wavelength) - wavelength))
	a_ph = a(None) - a_dg - absorb(wavelength) # differs from pdf doc; shown in spreadsheet	

	# Remove negatives
	b[b < 0] = 1e-5
	#a_ph[a_ph < 0] = 1e-5

	# QAA-CDOM - Zhu & Yu 2013
	a_p = 0.63 * b ** 0.88
	a_g = a(None) - absorb(key(443, wavelength)) - a_p  
	
	if lambda_reference is not None:
		return  idx(b)(lambda_reference),    \
				idx(a_ph)(lambda_reference), \
				idx(a_g)(lambda_reference), eta, S

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
	# return b + scatter(wavelength), a(None)



def QAA2(data, wavelength, lambda_reference=None):
	''' QAA v1 - https://pdfs.semanticscholar.org/c071/6255965a355245757132919a55956bab791f.pdf '''
	absorb, scatter = water_interpolators()
	wavelength = np.array(wavelength)

	# Functional interface into matrix, returning values in column with closest wavelength
	idx = lambda f: (lambda k: f[:, find(k, wavelength)][:, None] if k is not None else f)

	Rrs = idx( np.atleast_2d(data) )
	rrs = idx( to_rrs(Rrs(None)) )

	# Invert rrs formula to find u
	u  = idx( (-g0 + (g0**2 + 4 * g1 * rrs(None)) ** 0.5) / (2 * g1) )

	rho = np.log(rrs(440) / rrs(555))
	a440 = np.exp(-2. - 1.4 * rho + 0.2 * rho**2)
	a555 = 0.0596 + 0.2 * (a440 - 0.01)
	#a555 = 0.0596 + 0.56 * ((rrs(640) / rrs(555))**1.7 - 0.03)
	bbp555 = (u(555) * a555) / (1-u(555)) - scatter(555)
	Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * (rrs(440) / rrs(555))))
	bbp = bbp555 * (555 / wavelength) ** Y 
	a = idx(((1 - u(None)) * (scatter(wavelength) + bbp)) / u(None))
	zeta = 0.71 + (0.06 / (0.8 + rrs(440) / rrs(555)))
	xi = np.exp(0.015 * (440-410))
	ag440 = (a(410) - zeta * a(440)) / (xi - zeta) - (absorb(410) - zeta * absorb(440)) / (xi - zeta)
	aph440 = (a(440) - ag440 - absorb(440))
	ag = ag440 * np.exp(0.015 * (440 - wavelength))
	aph = a(None) - ag - absorb(wavelength)
	return {'aph': aph, 'ag': ag, 'a': a(None)}



def melin(source_data, source_wavelengths, target_wavelengths):
	''' Band adjustment - Melin & Sclep DOI: 10.1364/OE.23.002262 '''
	source_data = np.atleast_2d(source_data)
	source_wave = np.array(source_wavelengths)
	target_wave = np.array(target_wavelengths)
	assert(source_wave.shape[0] == source_data.shape[1]), \
		'Data / Wavelength mismatch: %s' % str([source_wave.shape[0], source_data.shape[1]])

	# Load Bricaud data - DOI: 10.1029/95JC00463
	root_dir = Path(__file__).parent
	AB_data  = np.loadtxt(root_dir.joinpath('IOP', 'AB_Bricaud.csv'), delimiter=',')
	A_interp = Akima(AB_data[:, 0], AB_data[:, 1])
	B_interp = Akima(AB_data[:, 0], AB_data[:, 2])

	lambda_reference = 443
	b, a_ph, a_g, eta, S = QAA(source_data, source_wave, lambda_reference)

	melin_out = []
	for lambda_target in target_wave:
		lambda_source_i  = [find(lambda_target, source_wave)]
		lambda_sources   = [source_wave[lambda_source_i[0]]]

		# Distance too great - use weighted average if not first / last index
		if abs(lambda_target - lambda_sources[0]) > 3 and lambda_source_i[0] not in [0, len(source_wave)-1]:
			lambda_source_i.append(lambda_source_i[0] + (1 if lambda_sources[0] < lambda_target else -1))
			lambda_sources.append(source_wave[lambda_source_i[1]])

		Rrs_es = []
		for lambda_source_idx, lambda_source in zip(lambda_source_i, lambda_sources):

			Rrs_fs = []
			for lmbda in [lambda_source, lambda_target]:
				bbp = b * (lambda_reference / lmbda) ** eta
				aph = A_interp(lmbda) * (a_ph / A_interp(lambda_reference)) ** ((1 - B_interp(lmbda)) / (1 - B_interp(lambda_reference)))
				acd = a_g * np.exp(-S * (lmbda - lambda_reference))
				
				rrs_f = g0 * (bbp / (bbp + (aph + acd))) + g1 * (bbp / (bbp + (aph + acd))) ** 2
				Rrs_fs.append( to_Rrs(rrs_f).flatten() )

			Rrs_source, Rrs_target = Rrs_fs
			Rrs_es.append( Rrs_target * (source_data[:, lambda_source_idx] / Rrs_source) )
		
		if len(lambda_sources) > 1:
			Rrs_e = np.abs(lambda_sources[1] - lambda_target) * Rrs_es[0] + np.abs(lambda_sources[0] - lambda_target) * Rrs_es[1]
			Rrs_e/= np.abs(lambda_sources[0] - lambda_sources[1])
		else:
			Rrs_e = Rrs_es[0]

		melin_p = Rrs_e
		melin_out.append(melin_p)

	return np.array(melin_out)


if __name__ == '__main__':
	Rrs = np.array([
		[0.0045, 0.0041, 0.00402, 0.00295, 0.00169, 0.00018],
		#[0.0012, 0.00169, 0.00329, 0.00404, 0.00748, 0.00346],
		#[0.00097, 0.00119, 0.00184, 0.00229, 0.00425, 0.00161],
	])
	wvl = np.array([412, 443, 490, 510, 555, 670])
	print(QAA(Rrs, wvl))