from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer
from collections import defaultdict as dd
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

from .plot_utils import add_stats_box, add_identity
from .benchmarks import bench_ml 
from .parameters import get_args 
from .metrics import mae, mape, rmsle, slope, msa, sspb
from .mdn2 import MDN 

import numpy as np 
import seaborn as sns


N_TRAIN  = 10000 
N_VALID  = 2000
N_TEST   = 20
N_SAMPLE = N_TRAIN + N_VALID + N_TEST


def add_noise(x_data, y_data, x_dep, x_ind, y_dep=None, y_ind=None):
	if y_dep is None: y_dep = x_dep
	if y_ind is None: y_ind = x_ind 

	# Generate noise
	x_dep_noise = x_dep * np.random.normal(size=x_data.shape) * x_data
	y_dep_noise = y_dep * np.random.normal(size=y_data.shape) * y_data
	x_ind_noise = x_ind * np.random.normal(size=x_data.shape) * np.abs(x_data).mean()
	y_ind_noise = y_ind * np.random.normal(size=y_data.shape) * np.abs(y_data).mean()
	return x_data+x_dep_noise+x_ind_noise, y_data+y_dep_noise+y_ind_noise


def split_and_process(x_data, y_data, x_orig, y_orig, n_train, n_test, scale=True):
	# Gather test, evenly across x space
	i_test = np.linspace(0, len(x_data)-1, n_test).astype(int)
	i_test = np.isin(np.arange(len(x_data)), i_test).astype(bool)
	x_test = x_data[i_test]
	y_test = y_data[i_test]

	# Gather remaining
	i_data = np.arange((~i_test).sum())
	np.random.shuffle(i_data)

	x_data = x_data[~i_test][i_data]
	y_data = y_data[~i_test][i_data]
	x_orig = x_orig[~i_test][i_data]
	y_orig = y_orig[~i_test][i_data]

	x_train = x_data[:n_train]
	y_train = y_data[:n_train]
	x_valid = x_data[n_train:]
	y_valid = y_data[n_train:]
	x_orig  = x_orig[n_train:]
	y_orig  = y_orig[n_train:]

	# Scale data
	if scale:
		sx = RobustScaler()
		sy = RobustScaler()
		sx.fit(x_train)
		sy.fit(y_train)

		x_train = sx.transform(x_train)
		y_train = sy.transform(y_train)
		x_valid = sx.transform(x_valid)
		y_valid = sy.transform(y_valid)
		x_test  = sx.transform(x_test)
		y_test  = sy.transform(y_test)
		x_orig  = sx.transform(x_orig)
		y_orig  = sy.transform(y_orig)
	return x_train, y_train, x_valid, y_valid, x_test, y_test, x_orig, y_orig


def get_data(dep_noise_pct, ind_noise_pct, minor_x=False, minor_y=False):
	def wave_function(y):
		# x = 7 * np.sin(2 * np.sin(0.75 * y) + y / 2)
		return 7 * np.sin(.75 * y) + y / 2

	# Generate data
	y_orig = np.random.uniform(-10, 10, (N_SAMPLE, 1))
	x_orig = wave_function(y_orig)

	# Sort by x
	x_orig, y_orig = map(np.array, zip(*sorted(zip(x_orig, y_orig), key=lambda k: k[0])))

	# Determine noise levels
	x_dep = y_dep = dep_noise_pct
	x_ind = y_ind = ind_noise_pct

	# If minor x noise: 5% dependent, 0% independent
	if minor_x:
		x_dep = 0.05
		x_ind = 0

	# If minor x noise: 0% dependent, 5% independent
	if minor_y:
		y_dep = 0
		y_ind = 0.05
	
	# Add noise
	x_data, y_data = add_noise(x_orig.copy(), y_orig.copy(), x_dep, x_ind, y_dep, y_ind)
	x_orig = x_data # X noise is always present

	x_train, y_train, x_valid, y_valid, x_test, y_test, x_orig, y_orig = \
		split_and_process(x_data, y_data, x_orig, y_orig, N_TRAIN, N_TEST)

	# Sort by x
	x_data, y_data = map(np.array, zip(*sorted(zip(x_data, y_data), key=lambda k: k[0])))
	x_orig, y_orig = map(np.array, zip(*sorted(zip(x_orig, y_orig), key=lambda k: k[0])))
	return x_train, y_train, x_valid, y_valid, x_test, y_test, x_orig, y_orig



if __name__ == '__main__':

	kwargs = {
		# 'n_mix'      : 10,
		# 'n_layers'   : 5,
		# 'n_hidden'   : 200,
		'n_iter'     : 10000,
		'n_redraws'  : 50,
		# 'batch': 256,
		# 'l2': 1e-5,
		# 'alpha': 1e-2,
		# 'lr': 1e-2,
		# 'independent_outputs' : True,
		'verbose': True,
	}
	kwargs = get_args(kwargs).__dict__
	kwargs['hidden'] = [kwargs['n_hidden']] * kwargs['n_layers']

	# Plot training progress
	if True:
		dep_noise_pct = 0.01
		ind_noise_pct = 0.01

		x_train, y_train, x_valid, y_valid, x_test, y_test, x_nonoise, y_nonoise = get_data(dep_noise_pct, ind_noise_pct)

		xmin, xmax = x_train.min()-abs(x_train.min()*0.1), x_train.max()*1.1
		ymin, ymax = y_train.min()-abs(y_train.min()*0.1), y_train.max()*1.1
		print(xmin, xmax, ymin, ymax)
		if 0:
			plt.plot(x_valid, y_valid, 'kx')
			plt.xlim((xmin, xmax))
			plt.ylim((ymin, ymax))
			plt.show()
			plt.plot(x_nonoise, y_nonoise, 'kx')
			plt.title('without noise')
			plt.show()

			# benchmarks = bench_ml(None, None, x_train[:1000], y_train[:1000], x_valid[:1000], y_valid[:1000], scale=False)
			benchmarks = bench_ml(None, None, x_train, y_train, x_valid, y_valid, scale=False, bagging=False)
			print('No noise:')
			bench_ml(None, None, x_train, y_train, x_nonoise, y_nonoise, scale=False, bagging=False)

			numrow = 2
			numcol = int(np.ceil(len(benchmarks)/numrow))
			get_xy = lambda n: (n%numrow, n//numrow)
			axes   = [plt.subplot2grid((numrow, numcol), get_xy(i)) for i in range(len(benchmarks))]
			
			for ax, (method, ests) in zip(axes, benchmarks.items()):
				ax.plot(y_valid, ests, 'bx')
				ax.set_xlim((xmin, xmax))
				ax.set_ylim((ymin, ymax))
				ax.set_title(method)
				add_stats_box(ax, y_valid, ests)
				add_identity(ax, color='k', ls='--')

			plt.show()

			axes = [plt.subplot2grid((numrow, numcol), get_xy(i)) for i in range(len(benchmarks))]
			for ax, (method, ests) in zip(axes, benchmarks.items()):
				ax.plot(x_valid, y_valid, 'kx')
				ax.plot(x_valid, ests, 'ro', alpha=0.5)
				ax.set_title(method)
				ax.set_xlim((xmin, xmax))
				ax.set_ylim((ymin, ymax))
				add_stats_box(ax, y_valid, ests)
			plt.show()


		# true_cov = np.cov(y_data.T, bias=1)
		# true_var = np.var(y_data, axis=0)

		# print('input 1:', x_data[0])
		# print('input 2:', x_data[1])
		# print('...')
		# print('output:', y_data[0])

		indices = np.arange(len(x_train))
		diffs   = []
		errs    = []
		losses  = []
		sums    = []
		mincov  = 1e10

		plt.figure(figsize=(20,4))
		plt.ion()
		ax  = plt.subplot2grid((1,5), (0,0))
		ax2 = plt.subplot2grid((1,5), (0,1))
		ax3 = plt.subplot2grid((1,5), (0,2))
		ax4 = plt.subplot2grid((1,5), (0,3))
		ax5 = plt.subplot2grid((1,5), (0,4))
		ax6 = ax5.twinx().twiny()

		ax.plot(x_train, y_train, 'kx')
		plt.show()
		plt.pause(1e-9)

		# from .mdn_ex import get_mdn, train_mdn, predict
		# model = get_mdn()

		model = MDN(**kwargs)
		model.n_in   = x_train.shape[1]
		model.n_pred = y_train.shape[1] 
		model.n_out  = model.n_mix * (1 + model.n_pred + (model.n_pred*(model.n_pred+1))//2) # prior, mu, (lower triangle) sigma
		model.construct_model()	

		likelihoods = np.zeros(len(x_train)) + 0.01
		picked = np.zeros(len(x_train))
		full_idxs = indices.copy()
		# np.random.shuffle(full_idxs)
		from scipy.special import softmax
		for it in trange(kwargs['n_iter']):

			if len(indices) < kwargs['batch']:
				indices = full_idxs.copy()
				np.random.shuffle(indices)
			
			# p = 1/likelihoods
			# # p = softmax(p)
			# p = p / p.sum()
			# idx = np.random.choice(full_idxs, kwargs['batch'], p=p)

			idx, indices = indices[:kwargs['batch']], indices[kwargs['batch']:]

			# _, loss, coefs, *mod_cov = model.session.run([model.train, model.loss, model.coefs],
			# 				feed_dict={model.x: x_train[idx], model.y: y_train[idx], model.is_training: True})
			# model = train_mdn(model, x_train, y_train)

			_, c = model.session.run([model.train, model.coefs], feed_dict={model.x: x_train[idx], model.y: y_train[idx], model.is_training: True})

			# likelihoods[idx] = np.max(c[0], 1)
			# picked[idx] += 1


			# prior : (n_sample, n_mix)
			# mu    : (n_sample, n_mix, n_out)
			# sigma : (n_sample, n_mix, n_out, n_out)
			# prior, mu, sigma = coefs 

			# top = prior == prior.max(1, keepdims=True)
			# top[top.sum(1) > 1] = np.eye(top.shape[1])[np.random.randint(top.shape[1])].astype(np.bool)

			# losses.append(abs(loss))
			# diffs.append((np.abs(mod_cov - true_cov)).mean(0).sum())
			# errs.append((np.abs(mu[top] - y_train[idx])).sum())
			# if it == 0:
			# 	print(sigma[top][:5])
			# 	print(corr[:5])
			# 	assert(0)
			# sums.append(mod_cov.sum())

			# if diffs[-1] < mincov:
			# 	mincov = diffs[-1]
			# 	tsig   = mod_cov
			
			if it % (kwargs['n_iter'] // kwargs['n_redraws']) == 0:
				ax5.cla()
				ax6.cla()
				# ax5.hist(likelihoods, bins=20)
				# ax6.plot(sorted(likelihoods), 'r')
				# print((picked == picked.min()).sum(), (picked == picked.max()).sum(), picked.min(), picked.max(), likelihoods[picked.argmax()], likelihoods[picked.argmin()], p[picked.argmin()], p[picked.argmax()])
				# ests = []
				# top = None
				# for _ in range(1):
				# 	prior, mu, sigma = model.session.run(model.coefs,
				# 			feed_dict={model.x: x_test})
				# 	if top is None:
				# 		top = prior == prior.max(1, keepdims=True)
				# 		top[top.sum(1) > 1] = np.eye(top.shape[1])[np.random.randint(top.shape[1])].astype(np.bool)
				# 	ests += [mu[top]]

				# scale_len = 1
				# drop_rate = 0
				# tau  = scale_len ** 2 * (1 - drop_rate) / (2*len(x_test)*model.scale_l2)
				# var  = np.var(ests,  0) + 1/tau
				# mean = np.mean(ests, 0)

				ax2.cla()
				# ax2.set_xlim((xmin, xmax))
				# ax2.set_ylim((ymin, ymax))
				# ax2.plot(x_train, y_train, 'kx', zorder=1)
				# ax2.plot(x_test, mean, 'r.', zorder=15)
				m_valid, valid_coef = model.session.run([model.most_likely, model.coefs], feed_dict={model.x: x_valid})

				# m_valid = MDN.get_most_likely_estimates( predict(model, x_valid) )

				# ax2.plot(x_valid, y_valid, 'b.', zorder=10, alpha=0.1)
				ax2.plot(y_valid, m_valid, 'bo', alpha=0.2)
				add_identity(ax2, color='k', ls='--')
				add_stats_box(ax2, y_valid, m_valid)


				# for x, m, v in zip(x_test, mean, var):
				# 	circle = Ellipse((x,m), 0.25, v+0.001)
				# 	circle.set_alpha(.5)
				# 	circle.set_facecolor('g')
				# 	circle.set_zorder(10)
				# 	ax2.add_artist(circle)

				prior, mu, sigma = model.session.run(model.coefs, feed_dict={model.x: x_test})
				# prior, mu, sigma = predict(model, x_test)

				ax.cla()
				ax.set_xlim((xmin, xmax))
				ax.set_ylim((ymin, ymax))

				ax.plot(x_train, y_train, 'kx', zorder=1)
				# ax.scatter(x_train.flatten(), y_train.flatten(), c=clusters/(clusters.max()+1), marker='x', alpha=0.5, zorder=1)

				top = prior == prior.max(1, keepdims=True)
				top[top.sum(1) > 1] = np.eye(top.shape[1])[np.random.randint(top.shape[1])].astype(np.bool)
				ax.plot(x_test, mu[top], 'r.', zorder=15)

				# def pdf(x, prior, mu, sigma):
				# 	val = 1
				# 	for m in range(prior.shape[1]):
				# 		mix_mu = mu[:, m]
				# 		mix_si = sigma[:, m]

				# 		k = mu.shape[1]
				# 		coef = ((2*np.pi)**k * np.det(mix_si)) ** -0.5
				valid_coef = model.session.run(model.coefs, feed_dict={model.x: x_train})

				valid_prior, valid_mu, valid_si = valid_coef
				valid_top = valid_prior.argmax(axis=1)

				# top_mu = valid_mu[np.arange(valid_mu.shape[0]), valid_top].flatten()
				# top_si = valid_si[np.arange(valid_mu.shape[0]), valid_top].flatten()

				# x, y, s = map(np.array, zip(*sorted(zip(x_valid.flatten(), top_mu, top_si), key=lambda z:z[1])))
				# print(x_valid[0].flatten(), top_mu[0], top_si[0])
				# s = 5 * s ** 2
				# ax.plot(x, y+s, color='g')
				# ax.plot(x, y, color='r')
				# ax.plot(x, y-s, color='g')


				def hessian(x):
					"""
					Calculate the hessian matrix with finite differences
					Parameters:
					   - x : ndarray
					Returns:
					   an array of shape (x.dim, x.ndim) + x.shape
					   where the array[i, j, ...] corresponds to the second derivative x_ij
					"""
					x_grad = np.gradient(x) 
					hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
					for k, grad_k in enumerate(x_grad):
						# iterate over dimensions
						# apply gradient again to every component of the first derivative.
						tmp_grad = np.gradient(grad_k) 
						for l, grad_kl in enumerate(tmp_grad):
							hessian[k, l, :, :] = grad_kl
					return hessian

				top_mu = valid_mu[np.arange(valid_mu.shape[0]), valid_top].flatten()
				top_si = valid_si[np.arange(valid_mu.shape[0]), valid_top]

				# We use the mixture mode as the mean, and calculate the resulting mixture covariance
				# print(np.transpose(valid_mu - top_mu[:,None,None], (0,2,1)).shape)
				# print(np.matmul(np.transpose(valid_mu - top_mu[:,None,None], (0,2,1)), valid_mu - top_mu[:,None, None]).shape)
				mixture_cov = valid_prior[..., None, None] * (valid_si + np.matmul(np.transpose(valid_mu - top_mu[:,None,None], (0,2,1)), valid_mu - top_mu[:,None, None])[:, None, ...])
				mixture_cov = mixture_cov.sum(axis=1)

				mixture_cov = top_si

				# print(mixture_cov.shape)
				# print(mixture_cov[0])
				# print()
				u, s, _ = np.linalg.svd(mixture_cov, hermitian=True)
				print(u.flatten())
				# print(u.shape, s.shape)
				# print(np.matmul(np.matmul(u, s[...,None]), np.transpose(u, (0,2,1)))[0])
				from scipy.special import erfinv

				# https://faculty.ucmerced.edu/mcarreira-perpinan/papers/cs-99-03.pdf
				# For a confidence level given by probability p (0<p<1) and number of dimensions d, rho is the error bar coefficient
				conf = 0.95#np.logspace(-1, 0, 30) - 1e-4 # 0.99
				# print(conf.min(), conf.max())
				rho = lambda p, d=1: 2 ** 0.5 * erfinv(p ** (1/d)) 

				# conf_surf = np.zeros((len(valid_si), len(conf)))
				# for ind in valid_prior.shape[1]:
				u, s, _ = np.linalg.svd(valid_si[:,ind], hermitian=True)
				bar = 2 * rho(conf) * s ** 0.5 # error bars centered at the mixture mode
					
					# conf_surf += bar * valid_prior[:, ind] 

				# print(bar.flatten())
				# print(bar.shape)
				# assert(0)

				x, y = map(np.array, zip(*sorted(zip(x_train.flatten(), top_mu), key=lambda z:z[1])))
				x2, y2 = map(np.array, zip(*sorted(zip(x_train.flatten(), top_mu+bar.flatten()), key=lambda z:z[1])))
				x3, y3 = map(np.array, zip(*sorted(zip(x_train.flatten(), top_mu-bar.flatten()), key=lambda z:z[1])))
				# print(list(zip(np.round(x,2), np.round(y,2))))
				# print(list(zip(np.round(x2,2), np.round(y2,2))))
				# print(list(zip(np.round(x3,2), np.round(y3,2))))

				ax.scatter(x2, y2, color='g', zorder=100)
				ax.scatter(x, y, color='r', zorder=100)
				ax.scatter(x3, y3, color='y', zorder=100)


				# assert(0)
				# ax.plot(x_valid, valid_coef[])
				for i,m in enumerate(mu[top]):

					circle = Ellipse((x_test[i], m), 0.25, sigma[top][i].flatten())#*np.diag(sigma[i,0]))
					circle.set_alpha(.8)
					circle.set_facecolor('g')
					circle.set_zorder(10)
					ax.add_artist(circle)

					# circle = Ellipse((x_test[i],m), 0.25, sigma[top][i].flatten() + var[i])
					# circle.set_alpha(.5)
					# circle.set_facecolor('cyan')
					# circle.set_zorder(9)
					# ax2.add_artist(circle)

				IDX   = 0
				sigma = sigma[..., IDX, IDX][None, ...] if len(sigma.shape) == 4 else sigma[..., IDX][None, ...]
				mu    = mu[..., IDX][None, ...]
				prior = prior[None, ...]
				Y   = np.linspace(y_train.min() * 1.5, y_train.max()*1.5, 300)[::-1, None, None]
				var = 2 * sigma ** 2

				num = np.exp(-(Y - mu) ** 2 / var)
				Z   = (prior * (num / (np.pi * var) ** 0.5)).sum(2)
				X, Y2 = np.meshgrid(x_test.flatten(), Y.flatten())
				# plt.contourf(X, Y2, MinMaxScaler((1e-3,1)).fit_transform(Z), norm=LogNorm(vmin=1e-3, vmax=1.), levels=np.logspace(-3, 0, 7), zorder=5, cmap='plasma', alpha=.1)

				ax.contour(X, Y2, MinMaxScaler((1e-3,1)).fit_transform(Z), norm=LogNorm(vmin=1e-3, vmax=1.), levels=np.logspace(-3, 0, 5), zorder=5, cmap='inferno', alpha=.5)
				# ax.scatter(x_test.flatten(), np.array([Y2[Z.argmax(0)[i], i] for i in range(Y2.shape[1])]), 3, label='Max Probability')
				# kde = sns.kdeplot(x_train.flatten(), y_train.flatten(), shade=False, ax=ax, bw='scott', n_levels=10, legend=False, gridsize=100, color='red')
				# kde.collections[2].set_alpha(0)

				if False:
					ax3.cla()
					ax3.plot(losses[::10])
					ax3.set_yscale('log')
				else:
					ax3.cla()

					(prior, mu, sigma), likely = model.session.run([model.coefs, model.most_likely], feed_dict={model.x: x_nonoise})
					# prior, mu, sigma = predict(model, x_nonoise)
					# likely = MDN.get_most_likely_estimates((prior, mu, sigma))

					add_stats_box(ax3, y_nonoise, likely)
					
					ax4.cla()
					# ax4.hist(np.max(prior, 1))
					ax4.hist(prior, stacked=True, bins=20)
					ax4.set_xlabel('Likelihood')
					ax4.set_ylabel('Frequency')
					
					# ax3.set_xlim((xmin, xmax))
					# ax3.set_ylim((ymin, ymax))
					# ax.plot(*y_data.T, 'kx')
					ax3.plot(x_nonoise, y_nonoise, 'kx', zorder=1)
					# ax.plot(*mu[:,0].T, 'r.')
					ax3.scatter(x_nonoise.flatten(), likely.flatten(), c=np.argmax(prior, 1).flatten()/prior.shape[1], marker='^', cmap='coolwarm', zorder=15, alpha=0.2)
					# print('diff:', np.abs(likely.flatten() - model.get_most_likely_estimates([prior, mu, sigma]).flatten()).sum())
					# ax3.scatter(x_nonoise.flatten(), model.get_most_likely_estimates([prior, mu, sigma]).flatten(), c=np.argmax(prior, 1).flatten()/prior.shape[1], cmap='jet', zorder=15, alpha=0.2)
					for m in range(mu.shape[1]):
						ax3.scatter(x_nonoise.flatten(), mu[:,m].flatten(), alpha=0.01)
					# ax3.scatter(x_nonoise.flatten(), np.sum(mu[...,0] * prior, 1).flatten())

					IDX   = 0
					sigma = sigma[..., IDX, IDX][None, ...] if len(sigma.shape) == 4 else sigma[..., IDX][None, ...]
					sigma = np.ones_like(sigma) * 0.5
					mu    = mu[..., IDX][None, ...]
					prior = prior[None, ...]
					Y   = np.linspace(y_nonoise.min() * 1.5, y_nonoise.max()*1.5, 1000)[::-1, None, None]
					# print(Y)
					num = np.exp(-0.5 * ((Y - mu) / sigma) ** 2)
					Z   = (prior * (num / (sigma * (2 * np.pi) ** 0.5) )).sum(2)
					# print(Z[:,0])
					# print(prior[:,0])
					# print(mu[:,0])
					# print(sigma[:,0])
					X, Y2 = np.meshgrid(x_nonoise.flatten(), Y.flatten())

					ax3.contourf(X, Y2, Z,  zorder=16, cmap='inferno', alpha=.3)
					# ax3.contour(X, Y2, MinMaxScaler((1e-3,1)).fit_transform(Z), norm=LogNorm(vmin=1e-3, vmax=1.), levels=np.logspace(-3, 0, 100), zorder=5, cmap='inferno', alpha=.5)
					# ax.scatter(x_test.flatten(), np.array([Y2[Z.argmax(0)[i], i] for i in range(Y2.shape[1])]), 3, label='Max Probability')


				plt.pause(1e-9)
				# input('?')

		input('finish?')
		print('True locs:', y_train[idx][:5])
		print('Est locs: ', mu[:5])
		print()
		print('True cov:   \n', true_cov, '\n')
		print('Closest cov:\n', tsig,     '\n')
		print('Current cov:\n', mod_cov,  '\n')
		print()
		print('True var:', true_var)
		print('Curr var:', np.var(var, axis=0))
		input()
		lines = []
		plt.ioff()
		plt.clf()
		lines += plt.plot(losses, label='Losses')
		plt.twinx()
		lines += plt.plot(diffs, 'g', label='Cov Errors')
		plt.yscale('log')
		plt.twinx()
		lines += plt.plot(errs, 'r', label='Loc Errors')
		plt.yscale('log')
		plt.twinx()
		lines += plt.plot(sums, 'orange', label='Cov Sums')
		plt.legend(lines, [l.get_label() for l in lines])
		plt.show()


	else:
		metrics = [(m, m.__name__.replace('MSA', 'Error').replace('SSPB','Bias')) for m in [msa, sspb]]
		metrics[1] = (lambda *args, **kwargs: np.abs(sspb(*args, **kwargs)), '|Bias|')
		n_trials= 20

		n_rows = 2
		n_cols = len(metrics) + 1

		dep_noise = [0]#np.linspace(0, 0.5, 2)
		ind_noise = np.linspace(0, 0.5, 11)

		stats_noise   = dd(lambda: dd(lambda: dd(lambda: dd(list))))
		stats_nonoise = dd(lambda: dd(lambda: dd(lambda: dd(list))))


		kwargs = {
			# 'n_mix'      : 10,
			# 'n_layers'   : 5,
			# 'n_hidden'   : 200,
			'n_iter'     : 10000,
			'no_bagging' : True,
			# 'n_redraws'  : 50,
			# 'batch': 256,
			# 'l2': 1e-5,
			# 'alpha': 1e-2,
			# 'lr': 1e-2,
			# 'independent_outputs' : True,
			'verbose': True,
		}

		kwargs['no_load'] = True 
		for dep_noise_pct in dep_noise:
			plt.figure(figsize=(4*n_cols,4*n_rows))
			plt.subplots_adjust(wspace=0.3)
			plt.ion()
			plt.show()
			axes = [[plt.subplot2grid((n_rows, n_cols), (i, j)) for j in range(n_cols)] for i in range(n_rows)]

			for ind_idx, ind_noise_pct in enumerate(ind_noise):

				noise_stats   = dd(list)
				nonoise_stats = dd(list)
				for _ in range(n_trials):
					# Gather estimates
					x_train, y_train, x_valid, y_valid, x_test, y_test, x_nonoise, y_nonoise = get_data(dep_noise_pct, ind_noise_pct)
					# x_train, y_train, x_valid, y_valid, x_test, y_test, x_nonoise, y_nonoise = get_data(ind_noise_pct, dep_noise_pct)
				
					benchmarks_noise, benchmarks_nonoise = bench_ml(None, None, x_train, y_train, x_valid, y_valid, x_other=x_nonoise, scale=False, bagging=False, silent=True, gridsearch=False)

					# kwargs['model_lbl'] = f'{dep_noise_pct}_{ind_noise_pct}_{_}'
					kwargs['no_save'] = True
					model = MDN(**kwargs)
					model.fit(x_train, y_train)
					# model.n_in   = x_train.shape[1]
					# model.n_pred = y_train.shape[1] 
					# model.n_out  = model.n_mix * (1 + model.n_pred + (model.n_pred*(model.n_pred+1))//2) # prior, mu, (lower triangle) sigma
					# model.construct_model()	

					# full_idx = np.arange(len(x_train))
					# indices  = []

					# for it in trange(kwargs['n_iter']):
					# 	if len(indices) < kwargs['batch']:
					# 		indices = full_idx.copy()
					# 		np.random.shuffle(indices)

					# 	idx, indices = indices[:kwargs['batch']], indices[kwargs['batch']:]
					# 	model.session.run(model.train, feed_dict={model.x: x_train[idx], model.y: y_train[idx], model.is_training: True})

					benchmarks_noise['MDN']   = model.predict(x_valid)#model.session.run(model.most_likely, feed_dict={model.x: x_valid})
					benchmarks_nonoise['MDN'] = model.predict(x_nonoise)#model.session.run(model.most_likely, feed_dict={model.x: x_nonoise})
					model.session.close()


					# Plot and store estimates
					for axs in axes:
						for ax in axs:
							ax.cla()
							ax.tick_params(labelsize=14)

					for i, (metric, name) in enumerate(metrics, 1):
						axes[0][i].set_title(name, fontsize=18)
						axes[1][i].set_xlabel('Noise', fontsize=18)
						axes[0][i].set_xticklabels([])
						axes[0][i].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
						axes[1][i].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
						axes[1][i].xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
					axes[0][0].set_ylabel(f'Validation With {ind_noise_pct*100:.0f}% Noise', fontsize=18)
					axes[1][0].set_ylabel('Validation Without Noise', fontsize=18)
					
					# axes[0][0].scatter(x_train, y_train, label='train')
					axes[0][0].scatter(x_valid, y_valid, label='valid')
					# axes[0][0].legend()
					axes[1][0].scatter(x_nonoise, y_nonoise)

					for method, estimates in benchmarks_noise.items():
						for i, (metric, name) in enumerate(metrics, 1):
							
							perf_w_noise  = metric(y_valid, estimates)
							perf_wo_noise = metric(y_nonoise, benchmarks_nonoise[method])

							stats_noise[name][method][dep_noise_pct][ind_noise_pct].append(perf_w_noise)
							stats_nonoise[name][method][dep_noise_pct][ind_noise_pct].append(perf_wo_noise)

							print(method, name, perf_w_noise, perf_wo_noise)

							for j, perf in enumerate([stats_noise, stats_nonoise]):
								val = perf[name][method][dep_noise_pct]
								avg = np.array([np.mean(val[k]) for k in ind_noise[:ind_idx+1]])
								std = np.array([np.std(val[k]) for k in ind_noise[:ind_idx+1]])

								ax = axes[j][i]
								ln = ax.plot(ind_noise[:ind_idx+1], avg)
								ax.scatter(ind_noise[:ind_idx+1], avg, label=method)
								ax.fill_between(ind_noise[:ind_idx+1], avg-std, avg+std, color=ln[0].get_color(), alpha=0.1)
							
								if j == 0:
									c  = ln[0].get_color()
									ax = axes[1][i]
									ax.plot(ind_noise[:ind_idx+1], avg, color=c, alpha=0.5, ls='--')
									# ax.scatter(ind_noise[:ind_idx+1], perf[metric.__name__][method][dep_noise_pct], color=c, alpha=0.2)

						print()
				
					# for i, (metric, name) in enumerate(metrics, 1):
					# 	axes[0][i].legend()
					# 	axes[1][i].legend()
					leg = axes[0][-1].legend(loc='lower right', bbox_to_anchor=(1.7,-0.5), fontsize=18)
					plt.pause(1e-8)

				print(f'\n{ind_noise_pct}\n------')
			print(f'\n{dep_noise_pct}\n------')
			# input('continue?')
			plt.savefig('toy_y.png', dpi=200, bbox_inches='tight', pad_inches=0.1, extra_artists=[leg])
			plt.ioff()
			plt.close()
			