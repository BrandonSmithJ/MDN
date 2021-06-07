# Product Estimation

### About
This repository contains source code for the following papers:

- <i>["Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach"](https://www.sciencedirect.com/science/article/pii/S0034425719306248). N. Pahlevan, et al. (2020). Remote Sensing of Environment. 111604. 10.1016/j.rse.2019.111604.</i>
- <i>["Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters"](https://www.sciencedirect.com/science/article/abs/pii/S0034425720301383). S.V. Balasubramanian, et al. (2020). Remote Sensing of Environment. 111768. 10.1016/j.rse.2020.111768.</i> [Code](https://github.com/BrandonSmithJ/MDN/tree/master/benchmarks/tss/SOLID).
- <i>["Hyperspectral retrievals of phytoplankton absorption and chlorophyll-a in inland and nearshore coastal waters"](https://www.sciencedirect.com/science/article/pii/S0034425720305733). N. Pahlevan, et al. (2021). Remote Sensing of Environment. 112200. 10.1016/j.rse.2020.112200.</i>
- <i>["A Chlorophyll-a Algorithm for Landsat-8 Based on Mixture Density Networks"](https://www.frontiersin.org/articles/10.3389/frsen.2020.623678/full). B. Smith, et al. (2021). Frontiers in Remote Sensing. 623678. 10.3389/frsen.2020.623678.</i>
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
sensor = "<OLI, MSI, OLCI, or HICO>"

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
bands, Rrs = get_tile_data("path/to/my/tile.nc", sensor, allow_neg=False) 
chla, idxs = image_estimates(Rrs, sensor=sensor)

# Or, with just random data:
import numpy as np 
random_data = np.random.rand(3, 3, len(get_sensor_bands(sensor)))
chla, idxs  = image_estimates(random_data, sensor=sensor)
```

Or, a .csv file may be given as input, with each row as a single sample. The .csv contents should be only the Rrs values to be estimated on (i.e. no header row or index column).

`python3 -m MDN --sensor <OLI, MSI, OLCI, or HICO> path/to/my/Rrs.csv`

*Note:* The user-supplied input values should correspond to R<sub>rs</sub> (units of 1/sr). 

Current performance is shown in the following scatter plots, with 50% of the data used for training and 50% for testing. Note that the models supplied in this repository are trained using 100% of the <i>in situ</i> data, and so observed performance may differ slightly. 

<p align="center">
	<img src=".res/S2B_benchmark.png?raw=true" height="311" width="721.5"></img>
	<br>
	<br>
	<img src=".res/OLCI_benchmark.png?raw=true" height="311" width="721.5"></img>
</p>



