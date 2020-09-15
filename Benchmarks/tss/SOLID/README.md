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
# Rrs = "<Your remote sensing reflectance [1/sr] data, shaped [N samples, N wavelengths]>"
# wavelengths = "<Wavelengths which are contained in the Rrs data>"
# For example:
from MDN.Benchmarks.tss.SOLID.model import model as SOLID
import numpy as np 
sensor = "OLI"
Rrs = np.random.random((6, 5)) # example data containing [6 samples, 5 wavelengths]
wavelengths = [443, 482, 561, 655, 865]
tss = SOLID(Rrs, wavelengths, sensor)
```

The previous SOLID version is provided at MDN/Benchmarks/tss/SOLID_old.zip. To use that version, simply unzip the folder into the same directory. We recommend using the latest version however, as the prior version is provided primarily for replicability.