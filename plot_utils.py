from .metrics import slope, sspb, mdsa 

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
	