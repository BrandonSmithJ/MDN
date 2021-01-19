from .. import image_estimates, get_tile_data, get_sensor_bands
import numpy as np

'''
All that's needed is to import image_estimates from MDN,
and pass in the Rrs data along with the sensor as a keyword.
For example:

> estimates = image_estimates(np.dstack([Rrs_443, Rrs_482, Rrs_561, Rrs_655]), sensor="OLI")
> chl = estimates[0]

Rrs bands should be passed in numerical order from least to 
greatest, and all bands with a wavelength < 800nm being passed 
in (with the exception of pan-chromatic bands). If uncertain,
the necessary bands for each sensor are listed in MDN/meta.py.

A few common ones are:
	OLI: [443, 482, 561, 655]
	S2B: [443, 490, 560, 665, 705, 740, 783]
	VI:  [410, 443, 486, 551, 671, 745] 
'''


TILES = {
	'S2B' : 'Tiles/S2B_T19LDC_20190206.nc',
	'OLI' : 'Tiles/OLI_l2gen.nc',
}


def random_data(sensor):	
	''' Estimate random data '''
	data = np.dstack([np.random.rand(3,3) for band in get_sensor_bands(sensor)])
	return image_estimates(data, sensor=sensor)


def estimate_tile(sensor, anc=False, rhos=False):
	''' Estimate actual data from a tile '''
	assert(sensor in TILES), f'No tile exists for sensor "{sensor}"'
	
	filename   = TILES[sensor]
	bands, Rrs = get_tile_data(filename, sensor, allow_neg=False, anc=anc, rhos=rhos)

	print(f'Tile shape: {Rrs.shape}')
	for band, vals in zip(bands, Rrs.T):
		n_valid = np.isfinite(vals).sum()
		rmin    = np.nanmin(vals)
		rmax    = np.nanmax(vals)
		print(f'Rrs{band} valid pixels: {n_valid:,}; Min: {rmin:.3f}; Max: {rmax:.3f}')	

	valid = np.all(np.isfinite(Rrs), -1)
	print(f'Total valid pixels: {valid.sum():,}')

	# DO NOT pass any model kwargs to image_estimates when using it in practice -
	# these being passed here should only be used when testing / debugging 
	model_kwargs = {
		'verbose': True,
		'n_rounds': 1,
	}

	chl = image_estimates(Rrs, anc=anc, rhos=rhos, sensor=sensor, **model_kwargs)[0]
	print(f'\nChl shape: {chl.shape}')
	print(f'Chl valid values: {np.isfinite(chl).sum()}')
	print(f'Max chl: {chl.max():.3f}  Min chl: {chl.min():.3f}  Avg chl: {chl.mean():.3f}')
	plot_estimates(chl)


def plot_estimates(chl):
	''' Show the estimated chl '''
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors

	cmin, cmax = chl.min(), chl.max()
	vmin = max(0, round(np.log10(cmin)))
	vmax = max(vmin+1, min(2, round(np.log10(cmax))))
	cmap = getattr(plt.cm, 'jet')
	cmap.set_bad(alpha=0)

	im = plt.imshow(chl, norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax), cmap=cmap)
	plt.colorbar(im)
	plt.show()



if __name__ == '__main__':
	for sensor in TILES:
		print(f'{sensor} chl:\n{random_data(sensor)}')	

	# sensor  = 'OLI'
	# estimate_tile(sensor, anc=True, rhos=True)	
