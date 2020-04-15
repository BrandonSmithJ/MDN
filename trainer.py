from .utils import get_labels, add_noise, add_identity, line_messages, find_wavelength
from .meta import get_sensor_bands
from .metrics import rmse, rmsle, mape, mae, leqznan
from .benchmarks import performance, bench_chl, bench_tss

from collections import defaultdict as dd
from sklearn import preprocessing
from pathlib import Path
from tqdm import trange
import numpy as np 
import warnings, os 


class DefaultArgs:
	verbose   = False
	plot_loss = False 
	animate   = False

class BatchIndexer:
	''' Returns minibatches of data for stochastic optimization. Allows 
		for data sampling bias via the prior probability output from MDN '''
	
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



def train_model(model, x_train, y_train, x_valid=None, y_valid=None, args=DefaultArgs()):
	plot_loss = args.plot_loss and x_valid is not None 

	# Create a live loss plot, which shows a large number of statistics 
	# during training. Show performance on the validation set as well.
	if plot_loss:
		import seaborn as sns 
		import matplotlib.pyplot as plt
		import matplotlib.animation as animation
		from matplotlib.patches import Ellipse
		from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec as GridSubplot

		train_losses = dd(list)
		valid_losses = dd(list)
		model_losses = []

		y_orig  = model.scalery.inverse_transform(y_train)
		x_valid = model.scalerx.transform(x_valid)
		
		train_val = np.append(x_train, x_valid, 0)	

		# Location of 0/-1 in the transformed space
		zero_line = model.scalery.inverse_transform(np.zeros((1, y_train.shape[-1])))
		neg_line  = model.scalery.inverse_transform(np.zeros((1, y_train.shape[-1]))-1)

		if args.darktheme: 
			plt.style.use('dark_background')

		cmap  = 'coolwarm'
		n_ext = 3 # extra rows, in addition to 1-1 scatter plots 
		n_col = min(5, y_valid.shape[1]) 
		n_row = n_ext + (n_col + n_col - 1) // n_col
		fig   = plt.figure(figsize=(5*n_col, 2*n_row))
		meta  = enumerate( GridSpec(n_row, 1, hspace=0.35) )
		conts = [GridSubplot(1, 2 if i in [0, n_row-1, n_row-2] else n_col, subplot_spec=o, wspace=0.3 if i else 0.45) for i, o in meta]
		axs   = [plt.Subplot(fig, sub) for container in conts for sub in container]
		axs   = axs[:n_col+2] + axs[-4:]
		[fig.add_subplot(ax) for ax in axs]
		axs   = [ax.twinx() for ax in axs[:2]] + axs  
		plt.ion()
		plt.show()
		plt.pause(1e-9)

		plot_metrics = [mape, rmsle]

		waves  = np.array(get_sensor_bands(args.sensor, args))
		labels = get_labels(waves, model.output_slices, n_col)[:n_col]

		if args.animate:
			ani_path = Path('Animations')
			ani_tmp  = ani_path.joinpath('tmp')
			ani_tmp.mkdir(parents=True, exist_ok=True)
			list(map(os.remove, ani_tmp.glob('*.png'))) # Delete any prior run temporary animation files
			
			# '-tune zerolatency' fixes issue where firefox won't play the mp4
			# '-vf pad=...' ensures height/width are divisible by 2 (required by .h264 - https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2) 
			extra_args = ["-tune", "zerolatency", "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white"]
			ani_writer = animation.writers['ffmpeg_file'](fps=3, extra_args=extra_args)
			ani_writer.setup(fig, ani_path.joinpath('MDN.mp4').as_posix(), dpi=100, frame_prefix=ani_tmp.joinpath('_').as_posix(), clear_temp=False)

	Batch = BatchIndexer(x_train, y_train, model.batch)
	for i in trange(model.n_iter, ncols=70, disable=not model.verbose):
		x_batch, y_batch = Batch.get_batch()

		# Add gaussian noise
		if (hasattr(args, 'no_noise') and not args.no_noise) or (hasattr(args, 'add_noise') and args.add_noise):
			x_batch, y_batch = add_noise(x_batch, y_batch, 0.02)

		*_, loss, (prior, mu, sigma) = model.session.run([model.train, model.loss, model.coefs], 
				feed_dict={model.x: x_batch, model.y: y_batch, model.is_training: True})

		Batch.update_stats(prior)
		
		# Update the performance log / plot
		if i and plot_loss and args.n_redraws > 0 and ((i+1) % (model.n_iter//args.n_redraws)) == 0:
			(prior, mu, sigma), est, avg = model.session.run([model.coefs, model.most_likely, model.avg_estimate], feed_dict={model.x: train_val})
			valid_loss = model.session.run(model.neg_log_pr, feed_dict={model.x: x_valid, model.y: model.scalery.transform(y_valid)})
			loss = model.session.run(model.neg_log_pr, feed_dict={model.x: x_train, model.y: y_train})

			est = model.scalery.inverse_transform(est)
			avg = model.scalery.inverse_transform(avg)

			train_est = est[:len(x_train)]
			train_avg = avg[:len(x_train)]
			valid_est = est[len(x_train):]
			valid_avg = avg[len(x_train):]

			for metric in plot_metrics:
				train_losses[metric.__name__].append([metric(y1, y2) for y1, y2 in zip(y_orig.T,  train_est.T)])
				valid_losses[metric.__name__].append([metric(y1, y2) for y1, y2 in zip(y_valid.T, valid_est.T)])
				
			model_losses.append([loss, leqznan(valid_est), valid_loss])
			valid_probs = np.max(prior, 1)[len(x_train):]
			valid_mixes = np.argmax(prior, 1)[len(x_train):]
			
			if model.verbose:
				line_messages([performance(  lbl, y1, y2) for lbl, y1, y2 in zip(labels, y_valid.T, valid_est.T)] + 
							  [performance('avg', y1, y2) for lbl, y1, y2 in zip(labels, y_valid.T, valid_avg.T)])

			if plot_loss:
				net_loss, zero_cnt, valid_loss = np.array(model_losses).T

				[ax.cla() for ax in axs]

				# Top two plots, showing training progress
				for axi, (ax, metric) in enumerate(zip(axs[:len(plot_metrics)], plot_metrics)):
					name = metric.__name__
					ax.plot(np.array(train_losses[name]), ls='--', alpha=0.5)
					ax.plot(np.array(valid_losses[name]), alpha=0.8)
					ax.set_ylabel(metric.__name__, fontsize=8)

					if axi==0: 
						ax.legend(labels, bbox_to_anchor=(1.2, 1 + .1*(y_train.shape[1]//6 + 1)), 
										  ncol=min(6, y_train.shape[1]), fontsize=8, loc='center')
				
				axi = len(plot_metrics)
				axs[axi].plot(net_loss, ls='--', color='w' if args.darktheme else 'k')
				axs[axi].plot(valid_loss, ls='--', color='gray')
				axs[axi].plot([np.argmin(valid_loss)], [np.min(valid_loss)], 'rx')
				axs[axi].set_ylabel('Network Loss', fontsize=8)
				axs[axi].tick_params(labelsize=8)
				axi += 1

				axs[axi].plot(zero_cnt, ls='--', color='w' if args.darktheme else 'k')
				axs[axi].set_ylabel('Est <= 0 Count', fontsize=8)
				axs[axi].tick_params(labelsize=8)
				axi += 1

				# Middle plots, showing 1-1 scatter plot estimates against measurements
				for yidx, lbl in enumerate(labels):
					ax   = axs[axi]
					axi += 1

					ax.scatter(y_valid[:, yidx], valid_est[:, yidx], 10, c=valid_mixes/prior.shape[1], cmap='jet', alpha=.5, zorder=5)
					ax.axhline(zero_line[0, yidx], ls='--', color='w' if args.darktheme else 'k', alpha=.5)
					# ax.axhline(neg_line[0, yidx], ls='-.', color='w' if args.darktheme else 'k', alpha=.5)
					add_identity(ax, ls='--', color='w' if args.darktheme else 'k', zorder=6)

					ax.tick_params(labelsize=5)
					ax.set_title(lbl, fontsize=8)

					with warnings.catch_warnings():
						warnings.filterwarnings('ignore')
						ax.set_xscale('log')
						ax.set_yscale('log')
						minlim = max(min(y_valid[:, yidx].min(), valid_est[:, yidx].min()), 1e-3)
						maxlim = min(max(y_valid[:, yidx].max(), valid_est[:, yidx].max()), 2000)
					
						if np.all(np.isfinite([minlim, maxlim])): 
							ax.set_ylim((minlim, maxlim)) 
							ax.set_xlim((minlim, maxlim))

					if (yidx % n_col) == 0:
						ax.set_ylabel('Estimate', fontsize=8)

					if (yidx // n_col) == (n_row-(n_ext+1)):
						ax.set_xlabel('Measurement', fontsize=8)

				# Bottom plot showing likelihood
				axs[axi].hist(valid_probs)
				axs[axi].set_xlabel('Likelihood')
				axs[axi].set_ylabel('Frequency')
				axi += 1

				axs[axi].hist(prior, stacked=True, bins=20)

				# Shows two dimensions of a few gaussians
				# circle = Ellipse((valid_mu[0], valid_mu[-1]), valid_si[0], valid_si[-1])
				# circle.set_alpha(.5)
				# circle.set_facecolor('g')
				# axs[axi].add_artist(circle)
				# axs[axi].plot([valid_mu[0]], [valid_mu[-1]], 'r.')
				# axs[axi].set_xlim((-2,2))#-min(valid_si[0], valid_si[-1]), max(valid_si[0], valid_si[-1])))
				# axs[axi].set_ylim((-2,2))#-min(valid_si[0], valid_si[-1]), max(valid_si[0], valid_si[-1])))

				# Bottom plot meshing together all gaussians into a probability-weighted heatmap
				# Sigmas are of questionable validity, due to data scaling interference
				with warnings.catch_warnings():
					warnings.filterwarnings('ignore')
					axi  += 1
					KEY   = list(model.output_slices.keys())[0]
					IDX   = model.output_slices[KEY].start
					sigma = sigma[len(x_train):, ...]
					sigma = model.scalery.inverse_transform(sigma.diagonal(0, -2, -1).reshape((-1, mu.shape[-1]))).reshape((sigma.shape[0], -1, sigma.shape[-1]))[..., IDX][None, ...]
					mu    = mu[len(x_train):, ...]
					mu    = model.scalery.inverse_transform(mu.reshape((-1, mu.shape[-1]))).reshape((mu.shape[0], -1, mu.shape[-1]))[..., IDX][None, ...]
					prior = prior[None, len(x_train):]

					Y   = np.logspace(np.log10(y_valid[:, IDX].min()*.5), np.log10(y_valid[:, IDX].max()*1.5), 100)[::-1, None, None]
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

					ydxs, ysort = np.array(sorted(enumerate(y_valid[:, IDX]), key=lambda v:v[1])).T
					Z    = Z[:, ydxs.astype(np.int32)]
					Ztop = Ztop[:, ydxs.astype(np.int32)]

					if np.any(np.isfinite(Ztop)):
						axs[axi].pcolormesh(np.arange(Z.shape[1]),Y,
							preprocessing.MinMaxScaler((0,1)).fit_transform(Ztop), cmap='inferno', shading='gouraud')				
					if np.any(np.isfinite(Z)):
						axs[axi].pcolormesh(np.arange(Z.shape[1]),Y, Z, cmap='BuGn_r', shading='gouraud', alpha=0.7)
					# axs[axi].colorbar()
					# axs[axi].set_yscale('symlog', linthreshy=y_valid[:, IDX].min()*.5)
					axs[axi].set_yscale('log')
					axs[axi].plot(ysort)#, color='red')
					axs[axi].set_ylabel(KEY)
					axs[axi].set_xlabel('in situ index (sorted by %s)' % KEY)
					axi += 1

					# Same as last plot, but only show the 20 most uncertain samples
					pc   = prior[0, ydxs.astype(np.int32)]
					pidx = np.argsort(pc.max(1))
					pidx = np.sort(pidx[:20])
					Z    = Z[:, pidx]
					Ztop = Ztop[:, pidx]
					if np.any(np.isfinite(Ztop)):
						axs[axi].pcolormesh(np.arange(Z.shape[1]),Y,
							preprocessing.MinMaxScaler((0,1)).fit_transform(Ztop), cmap='inferno')
					if np.any(np.isfinite(Z)):				
						axs[axi].pcolormesh(np.arange(Z.shape[1]),Y, Z, cmap='BuGn_r', alpha=0.7)

					axs[axi].set_yscale('log')
					axs[axi].plot(ysort[pidx])#, color='red')
					axs[axi].set_ylabel(KEY)
					axs[axi].set_xlabel('in situ index (sorted by %s)' % KEY)

				plt.pause(1e-9)

				# Store the current plot as a frame for the animation
				if len(model_losses) > 1 and args.animate:
					ani_writer.grab_frame()

					if ((len(model_losses) % 5) == 0) or ((i+1) == args.n_iter):
						ani_writer._run()

	# Move cursor to correct location
	if args.verbose: 
		if y_valid is not None:
			for _ in range(y_valid.shape[1] ): print()

	if args.animate:
		ani_writer.finish()

	# Allow choice of whether to save the current model
	if args.plot_loss:
		# input('continue?')
		plt.ioff()
		plt.close()