import matplotlib.pyplot as plt
import numpy as np 

from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
from mpl_toolkits.basemap import Basemap
from itertools import chain
from pathlib import Path 






def draw_map(*lonlats, scale=0.2, world=False, us=True, eu=False, labels=[], ax=None, gray=False, res='i', **scatter_kws):
	PLOT_WIDTH  = 8
	PLOT_HEIGHT = 6

	WORLD_MAP = {'cyl': [-90, 85, -180, 180]}
	US_MAP    = {
		'cyl' : [24, 49, -126, -65],
		'lcc' : [23, 48, -121, -64],
	}
	EU_MAP    = {
		'cyl' : [34, 65, -12, 40],
		'lcc' : [30.5, 64, -10, 40],
	}

	def mark_inset(ax, ax2, m, m2, MAP, loc1=(1, 2), loc2=(3, 4), **kwargs):
	    """
	    https://stackoverflow.com/questions/41610834/basemap-projection-geos-controlling-mark-inset-location
	    Patched mark_inset to work with Basemap.
	    Reason: Basemap converts Geographic (lon/lat) to Map Projection (x/y) coordinates

	    Additionally: set connector locations separately for both axes:
	        loc1 & loc2: tuple defining start and end-locations of connector 1 & 2
	    """
	    axzoom_geoLims = (MAP['cyl'][2:], MAP['cyl'][:2]) 
	    rect = TransformedBbox(Bbox(np.array(m(*axzoom_geoLims)).T), ax.transData)
	    pp   = BboxPatch(rect, fill=False, **kwargs)
	    ax.add_patch(pp)
	    p1 = BboxConnector(ax2.bbox, rect, loc1=loc1[0], loc2=loc1[1], **kwargs)
	    ax2.add_patch(p1)
	    p1.set_clip_on(False)
	    p2 = BboxConnector(ax2.bbox, rect, loc1=loc2[0], loc2=loc2[1], **kwargs)
	    ax2.add_patch(p2)
	    p2.set_clip_on(False)
	    return pp, p1, p2


	if world:
		MAP    = WORLD_MAP
		kwargs = {'projection': 'cyl', 'resolution': res}
	elif us:
		MAP    = US_MAP
		kwargs = {'projection': 'lcc', 'lat_0':30, 'lon_0':-98, 'resolution': res}#, 'epsg':4269}
	elif eu:
		MAP    = EU_MAP
		kwargs = {'projection': 'lcc', 'lat_0':48, 'lon_0':27, 'resolution': res}
	else:
		raise Exception('Must plot world, US, or EU')

	kwargs.update(dict(zip(['llcrnrlat', 'urcrnrlat', 'llcrnrlon', 'urcrnrlon'], MAP['lcc' if 'lcc' in MAP else 'cyl'])))
	if ax is None: f = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), edgecolor='w')
	m  = Basemap(ax=ax, **kwargs)
	ax = m.ax if m.ax is not None else plt.gca()

	if not world:
		m.readshapefile(Path(__file__).parent.joinpath('map_files', 'st99_d00').as_posix(), name='states', drawbounds=True, color='k', linewidth=0.5, zorder=11)
		m.fillcontinents(color=(0,0,0,0), lake_color='#9abee0', zorder=9)
		if not gray:
			m.drawrivers(linewidth=0.2, color='blue', zorder=9)
		m.drawcountries(color='k', linewidth=0.5)
	else:
		m.drawcountries(color='w')
	# m.bluemarble()
	if not gray:
		if us or eu: m.shadedrelief(scale=0.3 if world else 1)
		else:
			# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
			m.arcgisimage(service='World_Imagery', xpixels = 2000, verbose= True)
	else:
		pass
	# lats = m.drawparallels(np.linspace(MAP[0], MAP[1], 13))
	# lons = m.drawmeridians(np.linspace(MAP[2], MAP[3], 13))

	# lat_lines = chain(*(tup[1][0] for tup in lats.items()))
	# lon_lines = chain(*(tup[1][0] for tup in lons.items()))
	# all_lines = chain(lat_lines, lon_lines)
	
	# for line in all_lines:
	# 	line.set(linestyle='-', alpha=0.0, color='w')

	if labels:
		colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', 'xkcd:greyish blue', 'xkcd:goldenrod', ]
		markers = ['o', '^', 's', '*',  'v', 'X', '.', 'x',]
		mod_cr = False
		assert(len(labels) == len(lonlats)), [len(labels), len(lonlats)]
		for i, (label, lonlat) in enumerate(zip(labels, lonlats)):
			lonlat = np.atleast_2d(lonlat)
			if 'color' not in scatter_kws or mod_cr:
				scatter_kws['color'] = colors[i]
				scatter_kws['marker'] = markers[i]
				mod_cr = True
			ax.scatter(*m(lonlat[:,0], lonlat[:,1]), label=label, zorder=12, **scatter_kws)	
		ax.legend(loc='lower left', prop={'weight':'bold', 'size':8}).set_zorder(20)

	else:
		for lonlat in lonlats:
			if len(lonlat):
				lonlat = np.atleast_2d(lonlat)
				s = ax.scatter(*m(lonlat[:,0], lonlat[:,1]), zorder=12, **scatter_kws)
				# plt.colorbar(s, ax=ax)
	hide_kwargs = {'axis':'both', 'which':'both'}
	hide_kwargs.update(dict([(k, False) for k in ['bottom', 'top', 'left', 'right', 'labelleft', 'labelbottom']]))
	ax.tick_params(**hide_kwargs)

	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(1.5)
		ax.spines[axis].set_zorder(50)
	# plt.axis('off')

	if world:
		size = 0.35
		if us:
			loc = (0.25, -0.1) if eu else (0.35, -0.01)
			ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=loc, bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})
			
			scatter_kws.update({'s': 6})
			m2 = draw_map(*lonlats, labels=labels, ax=ax_ins, **scatter_kws)
			
			mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
			mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)


		if eu:
			ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=(0.75, -0.05), bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})
			
			scatter_kws.update({'s': 6})
			m2 = draw_map(*lonlats, us=False, eu=True, labels=labels, ax=ax_ins, **scatter_kws)
			
			mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
			mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)

	return m


if __name__ == '__main__':
	draw_map([-76, 37])
	plt.show()