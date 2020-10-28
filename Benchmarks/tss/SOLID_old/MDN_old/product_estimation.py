import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5' 

import tensorflow as tf
try: tf.logging.set_verbosity(tf.logging.ERROR)
except: tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np 
import pickle as pkl 
import argparse
import sys
import warnings
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from glob  import glob 
from tqdm  import trange 

PRODUCT_IDX = {
	'bb_p' : slice(0, 4 ),
	'a_p'  : slice(4, 8 ),
	'chl'  : slice(8, 9 ),
	'tss'  : slice(9, 10),
	'cdom' : slice(10,11),
	'eta'  : slice(11,12),
}

wavelengths = {
	'OLI'   : [443, 482, 561, 655],
	'MSI'   : [443, 490, 560, 665, 705],
	'OLCI'  : [411, 442, 490, 510, 560, 619, 664, 673, 681],
	'VI'    : [410, 443, 486, 551, 671], 
	'MOD'   : [412, 443, 488, 555, 667, 678],
}


SENSORS = ['OLI', 'VI', 'OLCI', 'MOD', 'MSI']

parser = argparse.ArgumentParser(epilog="""
	Passing a filename will estimate the desired parameter from the Rrs 
	contained in that file. Otherwise, a model will be trained (if not 
	already existing), and estimates will be made for the testing data.\n
""")
parser.add_argument("--verbose",   action ="store_true", help="Verbose output printing")
parser.add_argument("--max_prob",  action ="store_true", help="Use the maximum prior probability gaussian as the estimate. Otherwise, use weighted average")
parser.add_argument("--sensor",    default="OLI",        help="Sensor to estimate from.", choices=SENSORS)
parser.add_argument("--product",   default="bb_p",       help="Product to estimate. Currently only bbp is validated", choices=PRODUCT_IDX.keys())
parser.add_argument("--model_loc", default="Model",      help="Location of trained models")
parser.add_argument("--stratify",  action ="store_true", help="Stratified target variable sampling during training")
parser.add_argument("--seed",      default=None,   type=int,   help="Random seed")
parser.add_argument("--n_trials",  default=10,      type=int,   help="Number of models to fit, with median output as the final estimate")
parser.add_argument("--n_iter",    default=2001,  type=int,   help="Number of iterations to train the model")
parser.add_argument("--n_mix",     default=5,      type=int,   help="Number of gaussians to fit in the mixture model")
parser.add_argument("--batch",     default=100,    type=int,   help="Number of samples in a training batch")
parser.add_argument("--n_hidden",  default=100,    type=int,   help="Number of neurons per hidden layer")
parser.add_argument("--n_layers",  default=5,      type=int,   help="Number of hidden layers")
parser.add_argument("--n_redraws", default=100,    type=int,   help="Number of plot redraws during training (i.e. updates plot every n_iter / n_redraws iterations).")
parser.add_argument("--lr", 	   default=1e-3,   type=float, help="Learning rate")
parser.add_argument("--l1", 	   default=1e-4,   type=float, help="L1 normalization scale")
parser.add_argument("--l2", 	   default=1e-3,   type=float, help="L2 normalization scale")
parser.add_argument("--gradclip",  default=1e6,    type=float, help="Global norm scale for gradient clipping")


class TransformerPipeline(object):
	''' Apply multiple transformers seamlessly '''

	def __init__(self, scalers=None):
		if scalers is None: 	
			self.scalers = [
				FunctionTransformer(np.log, np.exp, validate=False),
				RobustScaler(),
				MinMaxScaler((0.01, 1)),
			]
		else:
			self.scalers = scalers 

	def fit(self, X):
		for scaler in self.scalers:
			X = scaler.fit_transform(X)

	def transform(self, X):
		for scaler in self.scalers:
			X = scaler.transform(X)
		return X

	def inverse_transform(self, X):
		for scaler in self.scalers[::-1]:
			X = scaler.inverse_transform(X)
		return X

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)


class CustomUnpickler(pkl.Unpickler):
	''' Ensure the classes are found, without requiring an import '''
	_warned = False

	def find_class(self, module, name):
		custom = {
			'TransformerPipeline' : TransformerPipeline,
			# 'neglog' : neglog,
			# 'neglog_inv' : neglog_inv,
		}
		if name in custom:
			return custom[name]
		return super().find_class(module, name)

	def load(self, *args, **kwargs):
		with warnings.catch_warnings(record=True) as w:
			pickled_object = super().load(*args, **kwargs)

		# For whatever reason, warnings does not respect the 'once' action for
		# sklearn's "UserWarning: trying to unpickle [...] from version [...] when
		# using version [...]". So instead, we catch it ourselves, and set the 
		# 'once' tracker via the unpickler itself.
		if len(w) and not CustomUnpickler._warned: 
			warnings.warn(w[0].message, w[0].category)
			CustomUnpickler._warned = True 
		return pickled_object

class MDN(object):
	@staticmethod
	def get_most_likely_estimates(coefs):
		''' Return mu for the distribution with the most likely prior '''
		mu = np.array(coefs[1])
		pr = np.array(coefs[0])
		return np.array([mu[s][np.argmax(pr,1).flatten()[s]] for s in np.arange(pr.shape[0])])

	@staticmethod
	def get_avg_estimates(coefs):
		''' Average mu for all distributions, weighted by their respective priors '''
		mu = np.array(coefs[1])
		pr = np.array(coefs[0])
		return np.array([np.sum(mu[s] * pr[s][:,None], 0) for s in np.arange(pr.shape[0])])


def create_model(args, model_path, model_uid, x_train=None, y_train=None, x_test=None, y_test=None, product_idx=None):
	checkpoint = tf.train.latest_checkpoint(model_path)
	tf.reset_default_graph()
	sess = tf.InteractiveSession()
	tf.set_random_seed(np.random.randint(1e10, dtype=np.int64))

	assert(checkpoint is not None), f'No model exists at {model_path}'
	if args.verbose: print("Restoring model weights from " + checkpoint)
	saver = tf.train.import_meta_graph(checkpoint + '.meta')
	graph = tf.get_default_graph()
	saver.restore(sess, checkpoint)

	# We want a copy of the class, without keeping defined attributes between trials
	class model(MDN): pass 
	model.x     = graph.get_tensor_by_name('x:0') 
	model.coefs = [graph.get_tensor_by_name('%s:0' % v) for v in ['prior', 'mu', 'sigma']] 

	with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
		scalerx, scalery, product_idx = CustomUnpickler(f).load()

	return model, sess, scalerx, scalery, product_idx



def estimate(args, x_train=None, y_train=None, x_test=None, y_test=None, product_idx=None):
	preds = []
	for trial in trange(args.n_trials, disable=not args.verbose or (args.n_trials == 1)):
		if args.seed is not None:
			np.random.seed(args.seed + trial)

		# Create the model (and train, if it doesn't already exist)
		root_dir    = os.path.abspath( os.path.dirname(__file__) )
		save_param  = ['sensor', 'n_mix', 'n_iter', 'batch', 'lr', 'l1', 'l2', 'gradclip', 'n_hidden', 'n_layers', 'stratify']
		model_id    = '_'.join(['%s%s' % (key.replace('_', ''), getattr(args, key)) for key in save_param])
		model_path  = os.path.join(root_dir, args.model_loc, model_id, 'trial%i' % trial)
		model_uid   = '%s__%s' % (model_id, 'trial%i' % trial)
		model, sess, scalerx, scalery, product_idx = create_model(
			args, model_path, model_uid, x_train, y_train, x_test, y_test, product_idx)

		# Apply trained model to the given test data
		partial = []
		for i in trange(0, len(x_test), args.batch*10, disable=not args.verbose):
			z_test = scalerx.transform(x_test[i:i+args.batch*10])
			coefs  = sess.run(model.coefs, feed_dict={model.x: z_test})
			if args.max_prob: estim = model.get_most_likely_estimates(coefs)
			else:             estim = model.get_avg_estimates(coefs)
			estim = scalery.inverse_transform(estim)
			partial.append(estim)

		preds.append(np.vstack(partial))
		sess.close()
	return preds, product_idx



def image_estimates(*bands, sensor='S2B', product_name='bb_p'):
	''' Takes any number of input bands (shaped [Height, Width]) and
		returns the products for that image, in the same shape. 
		Assumes the given bands are ordered from least to greatest 
		wavelength, and are the same bands used to train the network.
		Supported products: {bb_p, a_p, chl, tss, cdom}
	'''
	args = parser.parse_args([])
	args.sensor = sensor
	args.n_trials = 10
	args.product  = product_name
	assert(sensor in SENSORS), ('%s not in list of valid sensors: %s' % (sensor, SENSORS))
	assert(product_name in PRODUCT_IDX), (
		"Requested product unknown. Must be one of %s" % list(PRODUCT_IDX.keys()))
	assert(all([bands[0].shape == b.shape for b in bands])), (
		"Not all inputs have the same shape: %s" % str([b.shape for b in bands]))
	assert(len(bands) == len(wavelengths[args.sensor])), (
		"Got %s bands; expected %s bands for sensor %s" % (len(bands), len(wavelengths[args.sensor]), args.sensor))

	im_shape = bands[0].shape 
	im_data  = np.ma.vstack([b.flatten() for b in bands]).T
	im_mask  = np.any(im_data.mask, axis=1)
	im_data  = im_data[~im_mask]
	pred,idx = estimate(args, x_test=im_data)
	products = np.median(pred, 0) 
	product  = np.atleast_2d( products[:, idx[product_name]] )
	est_mask = np.tile(im_mask[:,None], (1, product.shape[1]))
	est_data = np.ma.array(np.zeros(est_mask.shape), mask=est_mask, hard_mask=True)
	est_data.data[~im_mask] = product
	return [p.reshape(im_shape) for p in est_data.T]
