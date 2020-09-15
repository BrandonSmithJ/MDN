from .metrics import slope, sspb, mdsa 
from .meta import get_sensor_label
from .utils import closest_wavelength, ignore_warnings
from collections import defaultdict as dd 
from pathlib import Path 
import numpy as np 


def add_identity(ax, *line_args, **line_kwargs):
	''' 
	Add 1 to 1 diagonal line to a plot.
	https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
	
	Usage: add_identity(plt.gca(), color='k', ls='--')
	'''
	line_kwargs['label'] = line_kwargs.get('label', '_nolegend_')
	identity, = ax.plot([], [], *line_args, **line_kwargs)
	
	def callback(axes):
		low_x, high_x = ax.get_xlim()
		low_y, high_y = ax.get_ylim()
		lo = max(low_x,  low_y)
		hi = min(high_x, high_y)
		identity.set_data([lo, hi], [lo, hi])

	callback(ax)
	ax.callbacks.connect('xlim_changed', callback)
	ax.callbacks.connect('ylim_changed', callback)

	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points', 
		'xycoords'   : 'axes fraction', 
		'fontname'   : 'monospace', 
		'xytext'     : (0,0), 
		'zorder'     : 25, 	
		'va'         : 'top', 
		'ha'         : 'left', 
	}
	ax.annotate(r'$\mathbf{1:1}$', xy=(0.87,0.99), size=11, **ann_kwargs)


def _create_metric(metric, y_true, y_est, longest=None, label=None):
	''' Create a position-aligned string which shows the performance via a single metric '''
	# if label == None:   label = metric.__name__.replace('SSPB', '\\beta').replace('MdSA', '\\varepsilon\\thinspace').replace('Slope','S\\thinspace')
	if label == None:   label = metric.__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')
	if longest == None: longest = len(label)

	ispct = metric.__qualname__ in ['mape', 'sspb', 'mdsa'] # metrics which are percentages
	diff  = longest-len(label)
	space = r''.join([r'\ ']*diff + [r'\thinspace']*diff)
	prec  = 1 if ispct else 3
	# prec  = 1 if abs(metric(y_true, y_est)) < 100 else 0
	stat  = f'{metric(y_true, y_est):.{prec}f}'
	perc  = r'$\small{\mathsf{\%}}$' if ispct else ''
	return rf'$\mathtt{{{label}}}{space}:$ {stat}{perc}'

def _create_stats(y_true, y_est, metrics, title=None):
	''' Create stat box strings for all metrics, assuming there is only a single target feature '''
	longest = max([len(metric.__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')) for metric in metrics])
	statbox = [_create_metric(m, y_true, y_est, longest=longest) for m in metrics]
	
	if title is not None:
		statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
	return statbox 

def _create_multi_feature_stats(y_true, y_est, metrics, labels=None):
	''' Create stat box strings for a single metric, assuming there are multiple target features '''
	if labels == None: 
		labels = [f'Feature {i}' for i in range(y_true.shape[1])]
	assert(len(labels) == y_true.shape[1] == y_est.shape[1]), f'Number of labels does not match number of features: {labels} - {y_true.shape}'
	
	title   = metrics[0].__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')
	longest = max([len(label) for label in labels])
	statbox = [_create_metric(metrics[0], y1, y2, longest=longest, label=lbl) for y1, y2, lbl in zip(y_true.T, y_est.T, labels)]
	statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
	return statbox 

def add_stats_box(ax, y_true, y_est, metrics=[mdsa, sspb, slope], bottom_right=False, x=0.025, y=0.97, fontsize=16, label=None):
	''' Add a text box containing a variety of performance statistics, to the given axis '''
	import matplotlib.pyplot as plt
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	create_box = _create_stats if len(y_true.shape) == 1 or y_true.shape[1] == 1 else _create_multi_feature_stats
	stats_box  = '\n'.join( create_box(y_true, y_est, metrics, label) )
	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points', 
		'xycoords'   : 'axes fraction', 
		'fontname'   : 'monospace', 
		'xytext'     : (0,0), 
		'zorder'     : 25, 	
		'va'         : 'top', 
		'ha'         : 'left', 
		'bbox'       : {
			'facecolor' : 'white',
			'edgecolor' : 'black', 
			'alpha'     : 0.7,
		}
	}

	ann = ax.annotate(stats_box, xy=(x,y), size=fontsize, **ann_kwargs)

	# Switch location to (approximately) the bottom right corner
	if bottom_right:
		plt.gcf().canvas.draw()
		bbox_orig = ann.get_tightbbox(plt.gcf().canvas.renderer).transformed(ax.transAxes.inverted())

		new_x = 1 - (bbox_orig.x1 - bbox_orig.x0) + x
		new_y = bbox_orig.y1 - bbox_orig.y0 + (1 - y)
		ann.set_x(new_x)
		ann.set_y(new_y)
		ann.xy = (new_x - 0.04, new_y + 0.06)
	return ann 
	

def draw_map(*lonlats, scale=0.2, world=False, us=True, eu=False, labels=[], ax=None, gray=False, res='i', **scatter_kws):
	''' Helper function to plot locations on a global map '''
	import matplotlib.pyplot as plt
	from matplotlib.transforms import Bbox
	from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
	from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
	from mpl_toolkits.basemap import Basemap
	from itertools import chain

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


def default_dd(d={}, f=lambda k: k):
	''' Helper function to allow defaultdicts whose default value returned is the queried key '''

	class key_dd(dd):
		''' DefaultDict which allows the key as the default value '''
		def __missing__(self, key):
			if self.default_factory is None:
				raise KeyError(key)
			val = self[key] = self.default_factory(key)
			return val 
	return key_dd(f, d)


@ignore_warnings
def plot_scatter(y_test, benchmarks, bands, labels, products, sensor):
	import matplotlib.patheffects as pe 
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt 
	import seaborn as sns 

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)

	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'aph' : '\\textit{a}_{ph}',
	})
	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')
	model_labels = default_dd({
		'MDN' : 'MDN_{A}',
	})

	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	# Only plot certain bands
	if len(labels) > 3 and 'chl' not in products:
		product_bands = {
			'default' : [443, 482, 561, 655],
			# 'aph'     : [443, 530],
		}

		target     = [closest_wavelength(w, bands) for w in product_bands.get(products[0], product_bands['default'])]
		plot_label = [w in target for w in bands]
		plot_order = ['QAA', 'GIOP']
		plot_bands = True
	else:
		plot_label = [True] * len(labels)
		plot_order = ['Smith_Blend', 'OC6', 'Mishra_NDCI', 'Gons_2band', 'Gilerson_2band']
		plot_bands = False

	labels = [(p,label) for label in labels for p in products if p in label]
	print('Plotting labels:', [l for i,l in enumerate(labels) if plot_label[i]])
	assert(len(labels) == y_test.shape[-1]), [len(labels), y_test.shape]

	plot_order = ['MDN'] + [p for p in plot_order if p in benchmarks]
	fig_size   = 5
	n_col      = max(3, sum(plot_label))
	n_row      = int(not plot_bands) + len(plot_order) // (1 if plot_bands else n_col)
	
	if plot_bands:
		n_col, n_row = n_row, n_col

	fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
	axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
	colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish']

	print('Order:', plot_order)
	print(f'Plot size: {n_row} x {n_col}')
	print(labels)

	curr_idx = 0
	full_ax  = fig.add_subplot(111, frameon=False)
	full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

	plabel = f'{product_labels[products[0]]} {product_units[products[0]]}'
	xlabel = fr'$\mathbf{{Measured {plabel}}}$'
	ylabel = fr'$\mathbf{{Modeled {plabel}}}$'
	full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
	full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)

	s_lbl = get_sensor_label(sensor).replace('-',' ')
	n_pts = len(y_test)
	title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
	full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)

	for plt_idx, (label, y_true) in enumerate(zip(labels, y_test.T)):
		if not plot_label[plt_idx]: continue 

		product, title = label 
		for est_idx, est_lbl in enumerate(plot_order):
			y_est = benchmarks[est_lbl][..., plt_idx]
			ax    = axes[curr_idx]
			cidx  = (curr_idx % n_col) if plot_bands else curr_idx
			color = colors[cidx]

			first_row = curr_idx < n_col #(curr_idx % n_row) == 0
			last_row  = curr_idx >= ((n_row-1)*n_col) #((curr_idx+1) % n_row) == 0
			first_col = (curr_idx % n_col) == 0
			last_col  = ((curr_idx+1) % n_col) == 0
		
			y_est_log  = np.log10(y_est).flatten()
			y_true_log = np.log10(y_true).flatten()
			curr_idx  += 1

			l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
			s_kws = {'alpha': 0.4, 'color': color}#, 'edgecolor': 'grey'}

			if est_lbl == 'MDN':
				[i.set_linewidth(5) for i in ax.spines.values()]
				est_lbl = 'MDN_{A}'
				est_lbl = 'MDN-I'
			else:
				est_lbl = est_lbl.replace('Mishra_','').replace('Gons_2band', 'Gons').replace('Gilerson_2band', 'GI2B').replace('Smith_','')

			if product not in ['chl', 'tss'] and last_col:
				ax2 = ax.twinx()
				ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
				ax2.grid(False)
				ax2.set_yticklabels([])
				ax2.set_ylabel(fr'$\mathbf{{{bands[plt_idx]:.0f}nm}}$', fontsize=20)

			minv = int(np.nanmin(y_true_log)) - 1 if product != 'aph' else -4
			maxv = int(np.nanmax(y_true_log)) + 1 if product != 'aph' else 1
			loc  = ticker.LinearLocator(numticks=maxv-minv+1)
			fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

			ax.set_ylim((minv, maxv))
			ax.set_xlim((minv, maxv))
			ax.xaxis.set_major_locator(loc)
			ax.yaxis.set_major_locator(loc)
			ax.xaxis.set_major_formatter(fmt)
			ax.yaxis.set_major_formatter(fmt)
			
			if not last_row:  ax.set_xticklabels([])
			if not first_col: ax.set_yticklabels([])

			valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
			if valid.sum():
				sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
				kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
				kde.collections[2].set_alpha(0)

			if len(valid.flatten()) != valid.sum():
				ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (~valid).sum())
				ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

			add_identity(ax, ls='--', color='k', zorder=20)
			add_stats_box(ax, y_true, y_est)

			if first_row or not plot_bands:
				ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$', fontsize=18)

			ax.tick_params(labelsize=18)
			ax.grid('on', alpha=0.3)

	filename = folder.joinpath(f'{products}_{sensor}_{n_pts}test.png')
	plt.tight_layout()
	# plt.subplots_adjust(wspace=0.35)
	plt.savefig(filename.as_posix(), dpi=100, bbox_inches='tight', pad_inches=0.1,)
	plt.show()
