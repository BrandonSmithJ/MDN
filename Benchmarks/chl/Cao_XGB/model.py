'''
XGBoost model of Cao et al. 2020
Requires Rrc - the rayleigh corrected reflectance, which is 
equivalent to rhos (surface reflectance) via SeaDAS.  
'''

from ...utils import get_required, optimize
from ..FAI.model import model as FAI 
from xgboost import DMatrix, Booster
from pathlib import Path 
import numpy as np

# Define any optimizable parameters
@optimize([])
def model(Rrc, wavelengths, *args, **kwargs):
	required = [443, 482, 561, 655, 865, 1610, 2200]
	tol = kwargs.get('tol', 10) # allowable difference from the required wavelengths
	Rrc_orig = Rrc.copy()
	Rrc = get_required(Rrc-Rrc[...,-1:], wavelengths, required, tol) # get values as a function: Rrc(443)
	bst = Booster()
	# bst.load_model(Path(__file__).parent.joinpath('model.json').as_posix())
	bst.load_model(Path(__file__).parent.joinpath('bst_oli', 'chl_bst_model_release.model').as_posix())

	features = [Rrc(w) for w in required[:-1]]
	features+= [Rrc(443) / Rrc(561)] # blue-green ratio index (Ha et al. 2017)
	features+= [Rrc(655) / Rrc(482)] # red-green ratio index (Watanabe et al. 2018)
	features+= [Rrc(865) / Rrc(655)] # infrared-red ratio index (Duan et al. 2007)
	features+= [FAI(Rrc(wavelengths), wavelengths, *args, **kwargs)] # FAI (Page et al. 2018)
	features = np.hstack(features)
	# assert(not np.any(np.isnan(features))), f'Indices with nan: {np.where(np.any(np.isnan(features), 0))}'
	return bst.predict(DMatrix(features))
