import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5' 
os.environ['TF_ENABLE_COND_V2'] = '1' 

import warnings
import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers.python import layers as tf_layers
from pathlib import Path 

from .utils import read_pkl, store_pkl
from .transformers import IdentityTransformer
from .trainer import train_model


class MDN(object):
	''' Mixture Density Network which handles multi-output, full (symmetric) covariance.

	Parameters
	----------
	n_mix : int, optional (default=5)
		Number of mixtures used in the gaussian mixture model.

	hidden : list, optional (default=[100, 100, 100, 100, 100])
		Number of layers and hidden units per layer in the neural network.

	lr : float, optional (default=1e-3)
		Learning rate for the model.

	l2 : float, optional (default=1e-3)
		L2 regularization scale for the model weights.

	n_iter : int, optional (default=1e5)
		Number of iterations to train the model for 

	batch : int, optional (default=128)
		Size of the minibatches for stochastic optimization.

	avg_est : bool, optional (default=False)
		If true, model outputs the prior probability weighted mean as the
		estimate. Otherwise, model outputs the maximum prior estimate.
	
	imputations : int, optional (default=5)
		Number of samples used in multiple imputation when handling NaN
		target values during training. More samples results in a higher
		accuracy for the likelihood estimate, but takes longer and may
		result in overfitting. Assumption is that any missing data is 
		MAR / MCAR, in order to allow a multiple imputation approach.

	epsilon : float, optional (default=1e-3)
		Normalization constant added to diagonal of the covariance matrix.

	threshold : float, optional (default=None)
		If set, the model outputs the maximum prior estimate when the prior
		probability is above this threshold; and outputs the average estimate
		when below the threshold. Any passed value should be in the range (0, 1],
		though the sign of the threshold can be negative in order to switch the
		estimates (i.e. negative threshold would output average estimate when prior
		is greater than the (absolute) value).  

	independent_outputs : bool, optional (default=False)
		Learn only the diagonal of the covariance matrix, such that 
		outputs have no covariance dependencies. 

	scalerx : transformer, optional (default=IdentityTransformer)
		Transformer which has fit, transform, and inverse_transform methods
		(i.e. follows the format of sklearn transformers). Scales the x 
		values prior to training / prediction. Stored along with the saved
		model in order to have consistent inputs to the model.

	scalery : transformer, optional (default=IdentityTransformer)
		Transformer which has fit, transform, and inverse_transform methods
		(i.e. follows the format of sklearn transformers). Scales the y 
		values prior to training, and the output values after prediction. 
		Stored along with the saved model in order to have consistent 
		outputs from the model.

	model_path : pathlib.Path, optional (default=./Model/)
		Folder location to store saved models.

	no_load : bool, optional (default=False)
		If true, train a new model rather than loading a previously 
		trained one.

	no_save : bool, optional (default=False)
		If true, do not save the model when training is completed.

	seed : int, optional (default=None)
		Random seed. If set, ensure consistent output.

	verbose : bool, optional (default=False)
		If true, print various information while loading / training.

	debug : bool, optional (default=False)
		If true, use control flow dependencies to determine where NaN
		values are entering the model. Model runs slower with this 
		parameter set to true.

	'''

	def __init__(self, n_mix=5, hidden=[100]*5, lr=1e-3, l2=1e-3, n_iter=1e5,
				 batch=128, avg_est=False, imputations=5, epsilon=1e-3,
				 threshold=None, independent_outputs=False, 
				 scalerx=IdentityTransformer(), scalery=IdentityTransformer(), 
				 model_path=Path('Model'), no_load=False, no_save=False,
				 seed=None, verbose=False, debug=False, **kwargs):

		self.n_mix        = n_mix
		self.hidden       = list(np.atleast_1d(hidden))
		self.lr           = lr
		self.l2           = l2
		self.n_iter       = n_iter
		self.batch        = batch
		self.avg_est      = avg_est
		self.imputations  = imputations
		self.eps          = epsilon
		self.threshold    = threshold
		self.distribution = 'MultivariateNormalDiag' if independent_outputs else 'MultivariateNormalFullCovariance'
		self.scalerx      = scalerx
		self.scalery      = scalery
		self.model_path   = model_path
		self.no_load      = no_load 
		self.no_save      = no_save
		self.seed         = seed 
		self.verbose      = verbose
		self.debug        = debug 

		self.graph   = tf.Graph()
		self.session = tf.compat.v1.Session(graph=self.graph)


	def fit(self, X, y, output_slices={'': slice(None)}, **kwargs):
		with self.graph.as_default():
			checkpoint = tf.train.latest_checkpoint(self.model_path)

			if checkpoint is not None and not self.no_load:
				if self.verbose: print("Restoring model weights from " + checkpoint)
				self.restore_model(checkpoint)

			elif self.no_load and X is None:
				raise Exception('Model exists, but no_load is set and no training data was given.')

			elif X is not None:
				self.output_slices = output_slices

				with warnings.catch_warnings():
					warnings.filterwarnings('ignore')
					X = self.scalerx.fit_transform(X)
					y = self.scalery.fit_transform(y)

				self.n_in   = X.shape[1]
				self.n_pred = y.shape[1] 
				self.n_out  = self.n_mix * (1 + self.n_pred + (self.n_pred*(self.n_pred+1))//2) # prior, mu, (lower triangle) sigma
		
				self.construct_model()
				train_model(self, X, y, **kwargs)
				self.save_model()

			else:
				raise Exception(f"No trained model exists at: \n{self.model_path}")
			self.graph.finalize()
		return self 


	def predict(self, X):
		with self.graph.as_default():
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore')
				target     = self.thresholded if self.threshold is not None else self.avg_estimate if self.avg_est else self.most_likely
				prediction = self.session.run(target, feed_dict={self.x: self.scalerx.transform(X), self.T: self.threshold})
				return self.scalery.inverse_transform(prediction)


	def construct_model(self):
		with self.graph.as_default():
			self.random = np.random.RandomState(self.seed)
			tf.compat.v1.set_random_seed(self.random.randint(1e10))

			self.global_step = tf.Variable(0, trainable=False, name='global_step')
			self.is_training = tf.compat.v1.placeholder_with_default(False, [], name='is_training')

			x = self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_in],   name='x')
			y = self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_pred], name='y')	
			T = self.T = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='T') 
			W = self.W = self.construct_weights()
			estimate   = self.forward(x, W)

			with tf.control_dependencies( self.debug_nan([estimate, x,]+[w[0] for mix_w in W for w in mix_w], names=['estim', 'x']+['w']*len([w[0] for mix_w in W for w in mix_w])) ):
				self.coefs = prior, mu, sigma = self.get_coefs(estimate)

			dist = getattr(tfd, self.distribution)(mu, sigma)
			prob = tfd.Categorical(probs=prior)
			mix  = tfd.MixtureSameFamily(prob, dist)

			def impute():
				return tf.reduce_mean([
					mix.log_prob( tf.compat.v2.where(tf.math.is_nan(y), mix.sample(), y) )
				for _ in range(self.imputations)], 0)

			likelihood = tf.compat.v2.cond(tf.reduce_any(tf.math.is_nan(y)), impute, lambda: mix.log_prob(y))
			neg_log_pr = tf.reduce_mean(-likelihood)
			l2_loss    = tf_layers.apply_regularization( tf_layers.l2_regularizer(scale=self.l2) )
			total_loss = neg_log_pr + l2_loss 

			self.neg_log_pr = neg_log_pr

			with tf.control_dependencies( tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) ):
				learn_rate  = self.lr 
				# learn_rate  = tf.train.polynomial_decay(self.lr, self.global_step, decay_steps=self.n_iter, end_learning_rate=self.lr/10)
				train_op    = tf.compat.v1.train.AdamOptimizer(learn_rate)
				grads, var  = zip(*train_op.compute_gradients(total_loss))

				with tf.control_dependencies( self.debug_nan(list(grads) + [total_loss], names=[v.name.split(':')[0] for v in var]+['loss']) ):
					self.train = train_op.apply_gradients(zip(grads, var), global_step=self.global_step, name='train_op')
					self.loss  = tf.identity(total_loss, name='model_loss')

			tf.compat.v1.global_variables_initializer().run(session=self.session)
			self.saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)


	def get_coefs(self, output):
		prior, mu, sigma = tf.split(output, [self.n_mix, self.n_mix*self.n_pred, -1], axis=1)

		with tf.control_dependencies( self.debug_nan([prior, mu, sigma], names=['prior', 'mu', 'sigma']) ):
			prior = tf.nn.softmax(prior, axis=-1) + 1e-9

			# Reshape tensors so that elements remain in the correct locations
			mu    = tf.stack(tf.split(mu, self.n_mix, 1), 1) 
			sigma = tf.stack(tf.split(sigma, self.n_mix, 1), 1) 
			sigma = tfd.fill_triangular(sigma, upper=False)

			# Explicitly set the shapes
			prior.set_shape((None, self.n_mix))
			mu.set_shape((None, self.n_mix, self.n_pred))
			sigma.set_shape((None, self.n_mix, self.n_pred, self.n_pred))

			# Independent outputs
			if self.distribution == 'MultivariateNormalDiag':
				sigma = tf.exp(tf.compat.v1.matrix_diag_part(sigma))
				norm  = tf.ones((1, 1, self.n_pred))

			# Full covariance estimation
			else:
				sigma = tf.einsum('abij,abjk->abik', tf.transpose(sigma, perm=[0,1,3,2]), sigma)
				norm  = tf.linalg.diag(tf.ones((1, 1, self.n_pred)))

			# Minimum uncertainty on covariance diagonal - prevents 
			# matrix inversion errors, and regularizes the model
			sigma+= self.eps * norm


			# _,var = tf.nn.moments(tf.stop_gradient(mu), [0])
			# var   = tf.abs(tf.tile(tf.expand_dims(tf.linalg.diag(var), 0), [tf.shape(sigma)[0], 1, 1, 1])) 
			# eps   = sigma * tf.reshape(tf.eye(self.n_pred), (1, 1, self.n_pred, self.n_pred))
			# sigma-= eps
			# sigma+= tf.clip_by_value(eps, var * 1e-3, np.inf) + 1e-8
			# sigma+= tf.clip_by_value(eps, tf.abs(tf.linalg.diag(mu)) * 1e-2, np.inf) + 1e-8
			# sigma += tf.abs(tf.linalg.diag(tf.stop_gradient(mu))) * 0.5


			# Store for model loading
			prior = tf.identity(prior, name='prior')
			mu    = tf.identity(mu,    name='mu')
			sigma = tf.identity(sigma, name='sigma')

			self.most_likely  = tf.identity(self.get_top(prior, mu), name='most_likely')
			self.avg_estimate = tf.identity(tf.reduce_sum(mu * tf.expand_dims(prior, -1), 1), name='avg_estimate') 
			self.thresholded  = tf.identity(tf.compat.v2.where(tf.expand_dims(tf.math.greater(tf.reduce_max(prior, 1) / self.T, tf.math.sign(self.T)), -1), self.most_likely, self.avg_estimate), name='thresholded')
			
			return prior, mu, sigma


	def restore_model(self, checkpoint):
		self.saver = tf.train.import_meta_graph(checkpoint + '.meta')
		self.saver.restore(self.session, checkpoint)
		
		self.x     = self.graph.get_tensor_by_name('x:0') 
		self.y     = self.graph.get_tensor_by_name('y:0') 
		self.T     = self.graph.get_tensor_by_name('T:0') 
		self.loss  = self.graph.get_tensor_by_name('model_loss:0')
		self.train = self.graph.get_operation_by_name('train_op')
		self.coefs = [self.graph.get_tensor_by_name('%s:0' % v) for v in ['prior', 'mu', 'sigma']] 
		self.avg_estimate = self.graph.get_tensor_by_name('avg_estimate:0')
		self.most_likely  = self.graph.get_tensor_by_name('most_likely:0')
		self.thresholded  = self.graph.get_tensor_by_name('thresholded:0')
		self.is_training  = self.graph.get_tensor_by_name('is_training:0')
		self.global_step  = self.graph.get_tensor_by_name('global_step:0')

		self.scalerx, self.scalery, self.output_slices, self.random = read_pkl(self.model_path.joinpath('scaler.pkl'))


	def save_model(self):
		if not self.no_save:
			self.model_path.mkdir(parents=True, exist_ok=True)
			store_pkl(self.model_path.joinpath('scaler.pkl'), [self.scalerx, self.scalery, self.output_slices, self.random])
			self.saver.save(self.session, self.model_path.joinpath('model'), global_step=self.global_step)


	def construct_weights(self):
		msra_init = lambda shape: tf_layers.xavier_initializer(uniform=True)(shape)
		create_W  = lambda shape, name: tf.Variable(msra_init(shape), name=name, collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tf.compat.v1.GraphKeys.WEIGHTS])
		create_b  = lambda shape, name: tf.Variable(msra_init(shape), name=name)
		create_Wb = lambda inp, out, i: (create_W([inp, out], f'layer{i}W'), create_b([out], f'layer{i}b'))

		in_sizes  = [self.n_in] + self.hidden
		out_sizes = self.hidden + [self.n_out]
		return [create_Wb(in_size, out_size, i)	for i, (in_size, out_size) in enumerate(zip(in_sizes, out_sizes))]
		

	def forward(self, inp, weights, funcs=[tf.nn.relu]):
		with tf.control_dependencies( self.debug_nan([inp], names=['input']) ):
			for i, (W, b) in enumerate(weights):
				if i: 
					for f in funcs: 
						inp = f(inp)
				inp = tf.matmul(inp, W) + b
			return inp


	def get_top(self, prior, values):
		''' Return values for the distribution with the most likely prior '''
		with tf.control_dependencies( self.debug_nan([prior, values], names=['prior', 'values']) ):
			vals, idxs  = tf.nn.top_k(prior, k=1)
			idxs = tf.stack([tf.range(tf.shape(idxs)[0]), tf.reshape(idxs, [-1])], axis=-1)
			return tf.gather_nd(values, idxs)


	def debug_nan(self, mats, names=[]):
		''' Create assertion dependencies for all given matrices, that all values are finite '''
		dependencies = []
		if self.debug:
			for i,mat in enumerate(mats):
				dependencies.append(tf.Assert(tf.reduce_all(tf.math.is_finite(mat)), [mat], name=names[i] if len(names) > i else '', summarize=1000))
		return dependencies
