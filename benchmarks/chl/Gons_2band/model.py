'''
Semi-analytical algorithm presented by Gons et al. (2002, 2005, 2008) 
which incorporates information on water absorption and backscattering
in relation to the MERIS red-NIR reflectance ratio
Using the definition found in Neil et al. 2019
'''

from ...utils import get_required, optimize
import numpy as np 

@optimize(['a', 'b'])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [665, 708, 778]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 1.063) # p
	b = kwargs.get('b', 0.016) # a*
	
	a_w665 = 0.4
	a_w708 = 0.7

	bb = 1.61 * np.pi * Rrs(778) / (0.082 - 0.6 * np.pi * Rrs(778))
	return ((Rrs(708) / Rrs(664)) * (a_w708 + bb) - a_w665 - bb**a) / b 
