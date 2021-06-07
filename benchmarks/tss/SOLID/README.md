# Statistical, inherent Optical property (IOP)-based, muLti-conditional Inversion proceDure (SOLID)

<i>"Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters". S.V. Balasubramanian, et al. (2020).</i>

### Usage
This package should first be cloned with the MDN library as a whole:

`git clone https://github.com/BrandonSmithJ/MDN`

<br>

SOLID may be used to estimate TSS for the missions:
- Landsat-8 (OLI)
- Sentinel-2 (MSI)
- Sentinel-3 (OLCI)
- Suomi-NPP (VI)
- Terra/Aqua (MOD)

The following code snippet shows how to use SOLID:
```
# sensor = "<OLI, MSI, VI, OLCI, MOD>"
# Rrs    = "<Your remote sensing reflectance [1/sr] data, shaped [N samples, N wavelengths]>"
# waves  = "<Wavelengths which are contained in the Rrs data>"
# For example:

from MDN.benchmarks.tss.SOLID.model import model as SOLID
import numpy as np 

sensor = "OLI"
waves  = [443, 482, 561, 655, 865]
Rrs    = np.random.random((6, len(waves)) # example data containing [6 samples, 5 wavelengths]
tss    = SOLID(Rrs, waves, sensor)

# Image estimates may also be generated via the following:
from MDN import image_estimates, get_tile_data
waves, Rrs = get_tile_data("path/to/my/tile.nc", sensor, allow_neg=False) 
tss, idxs  = image_estimates(Rrs, function=SOLID, sensor=sensor, wavelengths=waves)
```
