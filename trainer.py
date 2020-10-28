from .utils import get_labels, line_messages, find_wavelength, using_feature, ignore_warnings
from .meta import get_sensor_bands
from .metrics import rmse, rmsle, mape, mae, leqznan, sspb, mdsa
from .benchmarks import performance
from .plot_utils import add_identity

from collections import defaultdict as dd
from sklearn import preprocessing
from pathlib import Path
from tqdm import trange
import numpy as np 
import warnings, os, time


class DefaultArgs:
	verbose   = False
	plot_loss = False 
	animate   = False


class BatchIndexer:
	''' 
	Returns minibatches of data for stochastic optimization. Allows 
	for biased data sampling via the prior probability output from MDN.
	'''
	
	def __init__(self, X, y, batch, use_likelihood=False):
		self.X = X 
		self.y = y 
		self.batch   = batch
		self.indices = np.arange(len(X))
		self.current = []
		self.use_likelihood = use_likelihood
		self.likelihoods    = np.zeros(len(X)) + 0.01

	def get_batch(self):
		if self.use_likelihood:
			p = 1. / self.likelihoods
			self.idx = np.random.choice(self.indices, self.batch, p=p / p.sum())
		else:
			if len(self.current) < self.batch:
				self.current = self.indices.copy()
				np.random.shuffle(self.current)

			self.idx, self.current = self.current[:self.batch], self.current[self.batch:]
		return self.X[self.idx], self.y[self.idx]

	def update_stats(self, prior):
		if self.use_likelihood:
			self.likelihoods[self.idx] = np.max(prior, 1)


def add_noise(X, Y, percent=0.10):
	X += X * percent * np.random.normal(size=X.shape) + X * percent * np.random.choice([-1,1,0], size=(X.shape[0], 1))#(len(x_batch),1)) / 10 
	# Y += Y * percent * np.random.normal(size=Y.shape) + Y * percent * np.random.choice([-1,1,0], size=(Y.shape[0], 1))#(len(y_batch),1)) / 10
	return X, Y 


def save_training_results(args, model, data, i, start_time, first, metrics=[mdsa, sspb], folder='Results'):
	''' 
	Get estimates for the current iteration, applying the model to all available datasets. Store
	broad performance statistics, as well as the estimates for the first target feature.
	'''

	# Gather the necessary data into a single object in order to efficiently apply the model to all data at once
	all_keys = sorted(data.keys())
	all_data = [data[k]['x_t'] for k in all_keys]
	all_sums = np.cumsum(list(map(len, [[]] + all_data[:-1])))
	all_idxs = [slice(c, len(d)+c) for c,d in zip(all_sums, all_data)]
	all_data = np.vstack(all_data)

	# Create all estimates, transform back into original units, then split back into the original datasets
	estimates = model.session.run(model.most_likely, feed_dict={model.x: all_data})
	estimates = model.scalery.inverse_transform(estimates)
	estimates = {k: estimates[idxs] for k, idxs in zip(all_keys, all_idxs)}
	assert(all([estimates[k].shape == data[k]['y'].shape for k in all_keys])), \
		[(estimates[k].shape, data[k]['y'].shape) for k in all_keys]

	save_folder = Path(folder, args.config_name).resolve()
	if not save_folder.exists():
		print(f'\nSaving training results at {save_folder}\n')
		save_folder.mkdir(parents=True, exist_ok=True)

	# Save overall dataset statistics
	round_stats_file = save_folder.joinpath(f'round_{args.curr_round}.csv')
	if not round_stats_file.exists() or first:
		with round_stats_file.open('w+') as fn:
			fn.write(','.join(['iteration','cumulative_time'] + [f'{k}_{m.__name__}' for k in all_keys for m in metrics]) + '\n')

	stats = [[str(m(y1, y2)) for y1,y2 in zip(data[k]['y'].T, estimates[k].T)] for k in all_keys for m in metrics]
	stats = ','.join([f'[{s}]' for s in [','.join(stat) for stat in stats]])
	with round_stats_file.open('a+') as fn:
		fn.write(f'{i},{time.time()-start_time},{stats}\n')

	# Save model estimates
	save_folder = save_folder.joinpath('Estimates')
	if not save_folder.exists():
		save_folder.mkdir(parents=True, exist_ok=True)

	for k in all_keys:
		filename = save_folder.joinpath(f'round_{args.curr_round}_{k}.csv')
		if not filename.exists():
			with filename.open('w+') as fn:
				fn.write(f'target,{list(data[k]["y"][:,0])}\n')

		with filename.open('a+') as fn:
			fn.write(f'{i},{list(estimates[k][:,0])}\n')


class TrainingPlot:
	def __init__(self, args, model, data):
		self.args  = args 
		self.model = model
		self.data  = data

	def setup(self):
		self.train_test   = np.append(self.data['train']['x_t'], self.data['test']['x_t'], 0)	
		self.train_losses = dd(list)
		self.test_losses  = dd(list)
		self.model_losses = []

		# Location of 0/-1 in the transformed space
		self.zero_line = self.model.scalery.inverse_transform(np.zeros((1, self.data['train']['y_t'].shape[-1])))
		self.neg_line  = self.model.scalery.inverse_transform(np.zeros((1, self.data['train']['y_t'].shape[-1]))-1)

		if self.args.darktheme: 
			plt.style.use('dark_background')

		n_ext = 3 # extra rows, in addition to 1-1 scatter plots 
		n_col = min(5, self.data['test']['y'].shape[1]) 
		n_row = n_ext + (n_col + n_col - 1) // n_col
		fig   = plt.figure(figsize=(5*n_col, 2*n_row))
		meta  = enumerate( GridSpec(n_row, 1, hspace=0.35) )
		conts = [GridSubplot(1, 2 if i in [0, n_row-1, n_row-2] else n_col, subplot_spec=o, wspace=0.3 if i else 0.45) for i, o in meta]
		axs   = [plt.Subplot(fig, sub) for container in conts for sub in container]
		axs   = axs[:n_col+2] + axs[-4:]
		[fig.add_subplot(ax) for ax in axs]

		self.axes   = [ax.twinx() for ax in axs[:2]] + axs 
		self.labels = get_labels(get_sensor_bands(self.args.sensor, self.args), self.model.output_slices, n_col)[:n_col]
 
		plt.ion()
		plt.show()
		plt.pause(1e-9)

		if self.args.animate:
			ani_path = Path('Animations')
			ani_tmp  = ani_path.joinpath('tmp')
			ani_tmp.mkdir(parents=True, exist_ok=True)
			list(map(os.remove, ani_tmp.glob('*.png'))) # Delete any prior run temporary animation files
			
			# '-tune zerolatency' fixes issue where firefox won't play the mp4
			# '-vf pad=...' ensures height/width are divisible by 2 (required by .h264 - https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2) 
			extra_args = ["-tune", "zerolatency", "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white"]
			ani_writer = self.ani_writer = animation.writers['ffmpeg_file'](fps=3, extra_args=extra_args)
			ani_writer.setup(fig, ani_path.joinpath('MDN.mp4').as_posix(), dpi=100, frame_prefix=ani_tmp.joinpath('_').as_posix(), clear_temp=False)

	@ignore_warnings
	def update(self, plot_metrics=[mdsa, rmsle]):
		model = self.model
		if hasattr(model, 'session'):
			(prior, mu, sigma), est, avg = model.session.run([model.coefs, model.most_likely, model.avg_estimate], feed_dict={model.x: self.train_test})
			train_loss = model.session.run(model.neg_log_pr, feed_dict={model.x: self.data['train']['x_t'], model.y: self.data['train']['y_t']})				
			test_loss  = model.session.run(model.neg_log_pr, feed_dict={model.x: self.data['test' ]['x_t'], model.y: self.data['test' ]['y_t']})
		else:
			mix = model.model.layers[-1]
			tt_out = model.model(self.train_test)
			prior, mu, sigma = mix.get_coefs(tt_out)
			est = mix.get_most_likely(tt_out).numpy()
			avg = mix.get_avg_estimate(tt_out).numpy()
			train_loss = mix.loss(self.data['train']['y_t'], model.model(self.data['train']['x_t'])).numpy()
			test_loss  = mix.loss(self.data['test' ]['y_t'], model.model(self.data['test' ]['x_t'])).numpy()
			prior = prior.numpy()
			mu = mu.numpy()
			sigma = sigma.numpy()


		est = model.scalery.inverse_transform(est)
		avg = model.scalery.inverse_transform(avg)

		n_xtrain  = len(self.data['train']['x_t'])
		train_est = est[:n_xtrain ]
		train_avg = avg[:n_xtrain ]
		test_est  = est[ n_xtrain:]
		test_avg  = avg[ n_xtrain:]

		for metric in plot_metrics:
			self.train_losses[metric.__name__].append([metric(y1, y2) for y1, y2 in zip(self.data['train']['y'].T, train_est.T)])
			self.test_losses[ metric.__name__].append([metric(y1, y2) for y1, y2 in zip(self.data['test' ]['y'].T, test_est.T)])
			
		self.model_losses.append([train_loss, leqznan(test_est), test_loss])
		test_probs = np.max(   prior, 1)[n_xtrain:]
		test_mixes = np.argmax(prior, 1)[n_xtrain:]
		
		if model.verbose:
			line_messages([performance(  lbl, y1, y2) for lbl, y1, y2 in zip(self.labels, self.data['test']['y'].T, test_est.T)] + 
						  [performance('avg', y1, y2) for lbl, y1, y2 in zip(self.labels, self.data['test']['y'].T, test_avg.T)])

		net_loss, zero_cnt, test_loss = np.array(self.model_losses).T
		[ax.cla() for ax in self.axes]

		# Top two plots, showing training progress
		for axi, (ax, metric) in enumerate(zip(self.axes[:len(plot_metrics)], plot_metrics)):
			name = metric.__name__
			ax.plot(np.array(self.train_losses[name]), ls='--', alpha=0.5)
			ax.plot(np.array(self.test_losses[name]), alpha=0.8)
			ax.set_ylabel(metric.__name__, fontsize=8)

			if axi == 0: 
				n_targets = self.data['train']['y_t'].shape[1]
				ax.legend(self.labels, bbox_to_anchor=(1.2, 1 + .1*(n_targets//6 + 1)), 
									   ncol=min(6, n_targets), fontsize=8, loc='center')
		
		axi = len(plot_metrics)
		self.axes[axi].plot(net_loss, ls='--', color='w' if self.args.darktheme else 'k')
		self.axes[axi].plot(test_loss, ls='--', color='gray')
		self.axes[axi].plot([np.argmin(test_loss)], [np.min(test_loss)], 'rx')
		self.axes[axi].set_ylabel('Network Loss', fontsize=8)
		self.axes[axi].tick_params(labelsize=8)
		axi += 1

		self.axes[axi].plot(zero_cnt, ls='--', color='w' if self.args.darktheme else 'k')
		self.axes[axi].set_ylabel('Est <= 0 Count', fontsize=8)
		self.axes[axi].tick_params(labelsize=8)
		axi += 1

		# Middle plots, showing 1-1 scatter plot estimates against measurements
		for yidx, lbl in enumerate(self.labels):
			ax   = self.axes[axi]
			axi += 1

			ax.scatter(self.data['test']['y'][:, yidx], test_est[:, yidx], 10, c=test_mixes/prior.shape[1], cmap='jet', alpha=.5, zorder=5)
			ax.axhline(self.zero_line[0, yidx], ls='--', color='w' if self.args.darktheme else 'k', alpha=.5)
			# ax.axhline(neg_line[0, yidx], ls='-.', color='w' if self.args.darktheme else 'k', alpha=.5)
			add_identity(ax, ls='--', color='w' if self.args.darktheme else 'k', zorder=6)

			ax.tick_params(labelsize=5)
			ax.set_title(lbl, fontsize=8)			
			ax.set_xscale('log')
			ax.set_yscale('log')
			minlim = max(min(self.data['test']['y'][:, yidx].min(), test_est[:, yidx].min()), 1e-3)
			maxlim = min(max(self.data['test']['y'][:, yidx].max(), test_est[:, yidx].max()), 2000)
		
			if np.all(np.isfinite([minlim, maxlim])): 
				ax.set_ylim((minlim, maxlim)) 
				ax.set_xlim((minlim, maxlim))

			if yidx == 0:#(yidx % n_col) == 0:
				ax.set_ylabel('Estimate', fontsize=8)

			if yidx == 0:#(yidx // n_col) == (n_row-(n_ext+1)):
				ax.set_xlabel('Measurement', fontsize=8)

		# Bottom plot showing likelihood
		self.axes[axi].hist(test_probs)
		self.axes[axi].set_xlabel('Likelihood')
		self.axes[axi].set_ylabel('Frequency')
		axi += 1

		self.axes[axi].hist(prior, stacked=True, bins=20)

		# Shows two dimensions of a few gaussians
		# circle = Ellipse((valid_mu[0], valid_mu[-1]), valid_si[0], valid_si[-1])
		# circle.set_alpha(.5)
		# circle.set_facecolor('g')
		# self.axes[axi].add_artist(circle)
		# self.axes[axi].plot([valid_mu[0]], [valid_mu[-1]], 'r.')
		# self.axes[axi].set_xlim((-2,2))#-min(valid_si[0], valid_si[-1]), max(valid_si[0], valid_si[-1])))
		# self.axes[axi].set_ylim((-2,2))#-min(valid_si[0], valid_si[-1]), max(valid_si[0], valid_si[-1])))

		# Bottom plot meshing together all gaussians into a probability-weighted heatmap
		# Sigmas are of questionable validity, due to data scaling interference

		axi  += 1
		KEY   = list(model.output_slices.keys())[0]
		IDX   = model.output_slices[KEY].start
		sigma = sigma[n_xtrain:, ...]
		sigma = model.scalery.inverse_transform(sigma.diagonal(0, -2, -1).reshape((-1, mu.shape[-1]))).reshape((sigma.shape[0], -1, sigma.shape[-1]))[..., IDX][None, ...]
		mu    = mu[n_xtrain:, ...]
		mu    = model.scalery.inverse_transform(mu.reshape((-1, mu.shape[-1]))).reshape((mu.shape[0], -1, mu.shape[-1]))[..., IDX][None, ...]
		prior = prior[None, n_xtrain:]

		Y   = np.logspace(np.log10(self.data['test']['y'][:, IDX].min()*.5), np.log10(self.data['test']['y'][:, IDX].max()*1.5), 100)[::-1, None, None]
		var = 2 * sigma ** 2
		num = np.exp(-(Y - mu) ** 2 / var)
		Z   = (prior * (num / (np.pi * var) ** 0.5))
		I,J = np.ogrid[:Z.shape[0], :Z.shape[1]]
		mpr = np.argmax(prior, 2)
		Ztop= Z[I, J, mpr]
		Z[I, J, mpr] = 0
		Z   = Z.sum(2)
		Ztop += 1e-5
		Z    /= Ztop.sum(0)
		Ztop /= Ztop.sum(0)

		zp  = prior.copy()
		I,J = np.ogrid[:zp.shape[0], :zp.shape[1]]
		zp[I,J,mpr] = 0
		zp  = zp.sum(2)[0]
		Z[Z < (Z.max(0)*0.9)] = 0
		Z   = Z.T
		zi  = zp < 0.2
		Z[zi] = np.array([np.nan]*Z.shape[1])
		Z   = Z.T
		Z[Z == 0] = np.nan

		ydxs, ysort = np.array(sorted(enumerate(self.data['test']['y'][:, IDX]), key=lambda v:v[1])).T
		Z    = Z[:, ydxs.astype(np.int32)]
		Ztop = Ztop[:, ydxs.astype(np.int32)]

		if np.any(np.isfinite(Ztop)):
			self.axes[axi].pcolormesh(np.arange(Z.shape[1]),Y,
				preprocessing.MinMaxScaler((0,1)).fit_transform(Ztop), cmap='inferno', shading='gouraud')				
		if np.any(np.isfinite(Z)):
			self.axes[axi].pcolormesh(np.arange(Z.shape[1]),Y, Z, cmap='BuGn_r', shading='gouraud', alpha=0.7)
		# self.axes[axi].colorbar()
		# self.axes[axi].set_yscale('symlog', linthreshy=y_valid[:, IDX].min()*.5)
		self.axes[axi].set_yscale('log')
		self.axes[axi].plot(ysort)#, color='red')
		self.axes[axi].set_ylabel(KEY)
		self.axes[axi].set_xlabel('in situ index (sorted by %s)' % KEY)
		axi += 1

		# Same as last plot, but only show the 20 most uncertain samples
		pc   = prior[0, ydxs.astype(np.int32)]
		pidx = np.argsort(pc.max(1))
		pidx = np.sort(pidx[:20])
		Z    = Z[:, pidx]
		Ztop = Ztop[:, pidx]
		if np.any(np.isfinite(Ztop)):
			self.axes[axi].pcolormesh(np.arange(Z.shape[1]),Y,
				preprocessing.MinMaxScaler((0,1)).fit_transform(Ztop), cmap='inferno')
		if np.any(np.isfinite(Z)):				
			self.axes[axi].pcolormesh(np.arange(Z.shape[1]),Y, Z, cmap='BuGn_r', alpha=0.7)

		self.axes[axi].set_yscale('log')
		self.axes[axi].plot(ysort[pidx])#, color='red')
		self.axes[axi].set_ylabel(KEY)
		self.axes[axi].set_xlabel('in situ index (sorted by %s)' % KEY)

		plt.pause(1e-9)

		# Store the current plot as a frame for the animation
		if len(self.model_losses) > 1 and self.args.animate:
			ani_writer.grab_frame()

			if ((len(self.model_losses) % 5) == 0) or ((i+1) == int(self.args.n_iter)):
				ani_writer._run()


	def finish(self):
		if self.args.animate:
			ani_writer.finish()
		# input('continue?')
		plt.ioff()
		plt.close()



import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec as GridSubplot

# TODO: make the random state independent from the global random state (using seed from args)
def train_model(model, datasets, args=None):
	save_results = args is not None and 'test' in datasets and hasattr(args, 'save_stats') and args.save_stats
	plot_loss    = args is not None and 'test' in datasets and args.plot_loss

	# Create a live loss plot, which shows a large number of statistics during training
	if plot_loss:
		Plot = TrainingPlot(args, model, datasets)
		Plot.setup()

	start_time = time.time()
	Batch = BatchIndexer(datasets['train']['x_t'], datasets['train']['y_t'], model.batch)
	first = True

	for i in trange(int(model.n_iter), ncols=70, disable=not model.verbose):
		x_batch, y_batch = Batch.get_batch()

		# Add gaussian noise
		if args is not None and using_feature(args, 'noise'):
			x_batch, y_batch = add_noise(x_batch, y_batch, 0.02)

		*_, loss, (prior, mu, sigma) = model.session.run([model.train, model.loss, model.coefs], 
				feed_dict={model.x: x_batch, model.y: y_batch, model.is_training: True})

		Batch.update_stats(prior)
		
		if (plot_loss or save_results) and i and args.n_redraws > 0 and ((i+1) % (model.n_iter//args.n_redraws)) == 0:

			# Save performance to disk for later plotting (e.g. for learning curves)
			if save_results:
				save_training_results(args, model, datasets, i, start_time, first)
				first = False

			# Update the performance log / plot
			else: Plot.update()

	# Move cursor to correct location
	if args is not None and args.verbose: 
		if 'test' in datasets and datasets['test']['y'] is not None:
			for _ in range(datasets['test']['y'].shape[1] ): print()

	# Perform any plotting cleanup 
	if args is not None and args.plot_loss:
		Plot.finish()
