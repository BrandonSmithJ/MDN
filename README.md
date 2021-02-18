# Product Estimation

### About
This repository contains source code for the following papers:

- <i>["Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach"](https://www.sciencedirect.com/science/article/pii/S0034425719306248). N. Pahlevan, et al. (2020). Remote Sensing of Environment. 111604. 10.1016/j.rse.2019.111604.</i>
- <i>["Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters"](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301383). S.V. Balasubramanian, et al. (2020). Remote Sensing of Environment. 111768. 10.1016/j.rse.2020.111768.</i> [Code](https://github.com/BrandonSmithJ/MDN/tree/master/Benchmarks/tss/SOLID).
<br>

### Usage
The package can be cloned into a directory with:

`git clone https://github.com/BrandonSmithJ/MDN`

Alternatively, you may use pip to install:

`pip install git+https://github.com/BrandonSmithJ/MDN`

<br>

The code may then either be used as a library, such as with the following:
```
from MDN import image_estimates, get_tile_data, get_sensor_bands
sensor = "<S2A, S2B, or OLCI>"

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
bands, Rrs = get_tile_data("path/to/my/tile.nc", sensor, allow_neg=False) 

estimates = image_estimates(Rrs, sensor=sensor)
chlor_a   = estimates[0]

# Or, with just random data:
import numpy as np 
rand_data = np.dstack([np.random.rand(3,3) for band in get_sensor_bands(sensor)])
estimates = image_estimates(rand_data, sensor=sensor)
chlor_a   = estimates[0]
```

Or, a .csv file may be given as input, with each row as a single sample. The .csv contents should be only the Rrs values to be estimated on (i.e. no header row or index column).

`python3 -m MDN --sensor <S2A, S2B, or OLCI> path/to/my/Rrs.csv`

*Note:* The user-supplied input values should correspond to R<sub>rs</sub> (units of 1/sr). 

Current performance is shown in the following scatter plots, with 50% of the data used for training and 50% for testing. Note that the models supplied in this repository are trained using 100% of the <i>in situ</i> data, and so observed performance may differ slightly. 

<p align="center">
	<img src=".res/S2B_benchmark.png?raw=true" height="311" width="721.5"></img>
	<br>
	<br>
	<img src=".res/OLCI_benchmark.png?raw=true" height="311" width="721.5"></img>
</p>



