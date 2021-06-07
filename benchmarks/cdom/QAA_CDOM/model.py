'''
QAA CDOM @ 443nm
'''

from ...utils import get_required, optimize
from ...multiple.QAA.model import model as QAA

# Define any optimizable parameters
@optimize([])
def model(Rrs, wavelengths, *args, **kwargs):
	estimates = QAA(Rrs, wavelengths, *args, **kwargs)
	required  = [443]
	tol  = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	cdom = get_required(estimates['ag'], wavelengths, required, tol) # get values as a function: Rrs(443)
	return cdom(443)
