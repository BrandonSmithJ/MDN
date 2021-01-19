'''
Floating Algal Index (Hu, 2009)
Requires Rrc - the rayleigh corrected reflectance, which is 
equivalent to rhos (surface reflectance) via SeaDAS.  
'''

from ...utils import get_required, optimize, closest_wavelength
import numpy as np
# Define any optimizable parameters
@optimize([])
def model(Rrc, wavelengths, *args, **kwargs):
	required = [660, 860, 1600]
	tol = kwargs.get('tol', 20) # allowable difference from the required wavelengths
	Rrc = get_required(Rrc, wavelengths, required, tol) # get values as a function: Rrc(443)

	lambda_red  = closest_wavelength(660,  wavelengths, tol=tol)
	lambda_nir  = closest_wavelength(860,  wavelengths, tol=tol)
	lambda_swir = closest_wavelength(1600, wavelengths, tol=tol)

	red  = Rrc(lambda_red) 
	nir  = Rrc(lambda_nir)
	swir = Rrc(lambda_swir)

	nir_prime = red + (swir - red) * ((lambda_nir - lambda_red) / (lambda_swir - lambda_red))
	return nir - nir_prime