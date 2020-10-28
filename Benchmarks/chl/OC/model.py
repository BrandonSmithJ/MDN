'''
Function to call empirical OC Chl and Kd algorithms

Both algorithms have the form: 
	model = 10^(a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
where a0 ... a4 are polynomial regression coefficients,
x = log10(MBR), and MBR is maximum Rrs band ratio  

Usage:
	chl = get_oc([Rrs443,Rrs490,Rrs510,Rrs555], algorithm)
where Rrs is an array of Rrs at respective wavelengths 
and algorithm is 'OC4' or 'KD2S' or ... (see below)

Examples:
	for SeaWiFS Chl: oc4 = get_oc([Rrs443,Rrs490,Rrs510,Rrs555],'oc4')
	for MODIS Chl: oc3m = get_oc([Rrs443,Rrs488,-1,Rrs547],'oc3m')
	for SeaWiFS Kd: kd2s = get_oc([-1,Rrs490,-1,Rrs555],'kd2s')

Note: this function expects four Rrs arrays in a particular order:
blue (~443), blue (~488/490), green (~510/531), green (~547/555/560).
Use a placeholder for algorithms that dont use four Rrs, such as -1
in the above examples.  Some knowledge of what bands are used in each 
algorithm is therefore necessary,details provided here: 

http://oceancolor.gsfc.nasa.gov/ANALYSIS/ocv6/
http://oceancolor.gsfc.nasa.gov/ANALYSIS/kdv4/

https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
https://oceancolor.gsfc.nasa.gov/atbd/kd_490/

Implemented via matlab code by Jeremy Werdell, 2012
Brandon Smith, NASA Goddard Space Flight Center, April 2018
'''

from ...utils import get_required, optimize
from functools import partial, update_wrapper
import numpy as np 

OC_PARAM = {
	'OLI'  : 'L',
	'TM'   : 'L',
	'ETM'  : 'L',
	'ETM800'  : 'L',

	'MOD'  : 'M',
	'MODA' : 'M',
	'MODT' : 'M',

	'S2A'  : '-MSI',
	'S2B'  : '-MSI',
	'MSI'  : '-MSI',

	'OLCI' : '-S3A',

	'VI'   : 'V',

	'HICO' : 'H',
	'HYPER': 'H',

	'SEAWIFS' : 'S',
	'OCTS'    : 'O',
	'MERIS'   : 'E',
	'CZCS'    : 'C',
	'POLDER'  : 'P',
	'OSMI'    : 'I',
	'MOS'     : '-MOS',
}

# [coefficients, green band, blue band(s)] for the OC and KD algs
params = {
	# New coefficients (O'Reilly & Werdell, 2019)
	'OC6-S3A': ([ 0.2389,-1.9369,1.7627,-3.0777,-0.1054], [560, 665], [413, 443, 490, 510]), 
	'OC6-MOS': ([ 0.9541,-3.4581,2.9526,-1.3547, 0.0793], [570, 615], [408, 443, 485, 520]),
 	'OC6O'   : ([ 1.0597,-3.2499,2.4178,-1.1944, 0.1541], [565, 667], [412, 443, 490, 516]),
	'OC6E'   : ([ 0.9509,-3.0549,2.1814,-1.1178, 0.1513], [560, 665], [412, 442, 490, 510]),
	'OC6S'   : ([ 0.9216,-3.1788,2.3969,-1.3032, 0.2016], [555, 670], [412, 443, 490, 510]),
	'OC6M'   : ([ 1.2291,-4.9942,5.6471,-3.5343, 0.6927], [554, 667], [412, 442, 488, 531]),
	'OC6H'   : ([ 0.9618,-3.4379,2.8005,-1.5927, 0.2687], [553, 668], [416, 444, 490, 513]),
	'OC6I'   : ([ 0.9216,-3.1788,2.3969,-1.3032, 0.2016], [555, 670], [412, 443, 490, 510]),
	
	'OC5-S3A': ([ 0.4321,-3.1300,3.0548,-1.4518,-0.2495], [560], [413, 443, 490, 510]), 
	'OC5-MOS': ([ 0.6687,-3.6774,3.8455,-1.7762,-0.1377], [570], [408, 443, 485, 520]),
	'OC5O'   : ([ 0.5512,-3.4431,3.6141,-1.7857,-0.1520], [565], [412, 443, 490, 516]),
	'OC5E'   : ([ 0.4328,-3.1293,3.0487,-1.4348,-0.2547], [560], [412, 442, 490, 510]),
	'OC5S'   : ([ 0.3390,-3.1134,3.3570,-2.0179,-0.0381], [555], [412, 443, 490, 510]),
	'OC5M'   : ([ 0.4292,-4.8841,9.5768,-9.2429, 2.5192], [554], [412, 442, 488, 531]),
	'OC5H'   : ([ 0.3436,-3.4039,4.3482,-3.2685, 0.4155], [553], [416, 444, 490, 513]),
	'OC5I'   : ([ 0.3390,-3.1134,3.3570,-2.0179,-0.0381], [555], [412, 443, 490, 510]),

	'OC4-S3A': ([ 0.4254,-3.2168,2.8691,-0.6263,-1.0934], [560], [443, 490, 510]), 
	'OC4-MOS': ([ 0.6632,-3.7590,3.6769,-1.0312,-0.8426], [570], [443, 485, 520]),
	'OC4M'   : ([ 0.2702,-2.4794,1.5375,-0.1397,-0.6617], [554], [412, 442, 488]),
	'OC4H'   : ([ 0.3353,-3.4869,4.2086,-2.6434,-0.3555], [553], [444, 490, 513]),
	'OC4I'   : ([ 0.3281,-3.2073,3.2297,-1.3677,-0.8174], [555], [443, 490, 510]),
	'OC4V'   : ([ 0.2610,-2.5397,1.6345,-0.2116,-0.6655], [551], [410, 443, 486]),

	'OC3-S3A': ([ 0.3308,-2.6684,1.5990, 0.5525,-1.4876], [560], [443, 493]), 
	'OC3-MSI': ([ 0.3308,-2.6684,1.5990, 0.5525,-1.4876], [560], [443, 493]),      # S3A duplicate
	'OC3H'   : ([ 0.3308,-2.6684,1.5990, 0.5525,-1.4876], [560], [443, 493]),      # S3A duplicate
	'OC3P'   : ([ 0.4171,-2.5640,1.2222, 1.0275,-1.5680], [565], [443, 490]),

	'OC2-S3A': ([ 0.2389,-1.9369,1.7627,-3.0777,-0.1054], [560], [490]),     
	'OC2-MSI': ([ 0.2389,-1.9369,1.7627,-3.0777,-0.1054], [560], [490]),           # S3A duplicate
	'OC2H'   : ([ 0.2389,-1.9369,1.7627,-3.0777,-0.1054], [560], [490]),           # S3A duplicate
	'OC2P'   : ([ 0.1987,-1.7830,0.8457, 0.1946,-0.9563], [565], [443]),


	# Operational coefficients
	'OC4E'   : ([ 0.3255,-2.7677,2.4409,-1.1288,-0.4990], [560], [443, 490, 510]), #  MERIS operational Chl
	'OC4O'   : ([ 0.3325,-2.8278,3.0939,-2.0917,-0.0257], [565], [443, 490, 516]), #  OCTS operational Chl
	'OC4S'   : ([ 0.3272,-2.9940,2.7218,-1.2259,-0.5683], [555], [443, 490, 510]), #  SeaWiFS operational Chl
	'OC4'    : ([ 0.3272,-2.9940,2.7218,-1.2259,-0.5683], [555], [443, 490, 510]), #  SeaWiFS operational Chl

	'OC3S'   : ([ 0.2515,-2.3798,1.5823,-0.6372,-0.5692], [555], [443, 490]), #  SeaWiFS 3 band Chl
	'OC3E'   : ([ 0.2521,-2.2146,1.5193,-0.7702,-0.4291], [560], [443, 486]), #  MERIS 3 band Chl
	'OC3O'   : ([ 0.2399,-2.0825,1.6126,-1.0848,-0.2083], [565], [443, 490]), #  OCTS 3 band Chl
	'OC3M'   : ([ 0.2424,-2.7423,1.8017, 0.0015,-1.2280], [547], [443, 488]), #  MODIS operational Chl
	'OC3V'   : ([ 0.2228,-2.4683,1.5867,-0.4275,-0.7768], [550], [443, 487]), #  VIIRS operational Chl
	'OC3C'   : ([ 0.3330,-4.3770,7.6267,-7.1457, 1.6673], [550], [443, 520]), #  CZCS operational Chl
	'OC3L'   : ([ 0.2412,-2.0546,1.1776,-0.5538,-0.4570], [561], [443, 482]), #  OLI / Landsat 8 Chl
	'OC3'    : ([ 0.2412,-2.0546,1.1776,-0.5538,-0.4570], [561], [443, 482]), #  OLI / Landsat 8 Chl

	'OC2S'   : ([ 0.2511,-2.0853,1.5035,-3.1747, 0.3383], [555], [490]), #  SeaWiFS 2 band Chl
	'OC2E'   : ([ 0.2389,-1.9369,1.7627,-3.0777,-0.1054], [560], [490]), #  MERIS 2 band Chl
	'OC2O'   : ([ 0.2236,-1.8296,1.9094,-2.9481,-0.1718], [565], [490]), #  OCTS 2 band Chl
	'OC2M'   : ([ 0.2500,-2.4752,1.4061,-2.8237, 0.5405], [547], [488]), #  MODIS 2 band Chl
	'OC2M-HI': ([ 0.1464,-1.7953,0.9718,-0.8319,-0.8073], [555], [469]), #  MODIS high-res band Chl
	'OC2V'   : ([ 0.2230,-2.1807,1.4434,-3.1709, 0.5863], [550], [487]), #  VIIRS 2 band Chl
	'OC2L'   : ([ 0.1977,-1.8117,1.9743,-2.5635,-0.7218], [561], [482]), #  OLI / Landsat 8 Chl
	'OC2'    : ([ 0.1977,-1.8117,1.9743,-2.5635,-0.7218], [561], [482]), #  OLI / Landsat 8 Chl

	'KD2S'   : ([-0.8515,-1.8263,1.8714,-2.4414,-1.0690], [555], [490]), # SeaWiFS operational Kd
	'KD2E'   : ([-0.8641,-1.6549,2.0112,-2.5174,-1.1035], [560], [490]), # MERIS operational Kd
	'KD2O'   : ([-0.8878,-1.5135,2.1459,-2.4943,-1.1043], [565], [490]), # OCTS operational Kd
	'KD2M'   : ([-0.8813,-2.0584,2.5878,-3.4885,-1.5061], [547], [488]), # MODIS operational Kd
	'KD2V'   : ([-0.8730,-1.8912,1.8021,-2.3865,-1.0453], [550], [490]), # VIIRS operational Kd
	'KD2C'   : ([-1.1358,-2.1146,1.6474,-1.1428,-0.6190], [520], [443]), # CZCS operational Kd
	'KD2L'   : ([-0.9054,-1.5245,2.2392,-2.4777,-1.1099], [561], [482]), # OLI / Landsat 8 Kd
}

def OC(Rrs, wavelengths, sensor, *args, algorithm='OC', num=3, **kwargs):
	name = f'{algorithm}{num}{OC_PARAM[sensor] if sensor else ""}'.upper()
	assert(name in params), 'Unknown algorithm "%s".' % name
	
	defaults, greens, blues = params[name]
	req  = sorted(list(set(greens + blues)))
	tol  = kwargs.get('tol', 9)
	Rrs  = get_required(Rrs, wavelengths, req, tol)
	coef = [kwargs.get(k, d) for k, d in zip('abcde', defaults)]
	var  = np.log10(Rrs(blues).max(axis=1, keepdims=True) / Rrs(greens).mean(axis=1, keepdims=True))
	val  = 10 ** ((np.atleast_2d(coef) * var ** np.arange(5)).sum(axis=1, keepdims=True))

	# K_bio -> Kd
	if name[0] == 'K':
		val += 0.0166
	return val

def OC_factory(num):
	func = update_wrapper(partial(OC, num=num), OC)
	func = optimize(['a', 'b', 'c', 'd', 'e'])(func)
	func.model_name = f'OC{num}' # Cannot set __name__ directly as most models use the folder as their label
	return func

model2 = OC_factory(2)
model3 = OC_factory(3)
model4 = OC_factory(4)
model5 = OC_factory(5)
model6 = OC_factory(6)
