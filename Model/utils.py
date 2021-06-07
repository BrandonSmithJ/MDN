import numpy as np 
import tensorflow as tf 


def initialize_random_states(seed=None):
	''' Initialize the numpy and tensorflow random states, setting the tensorflow global random state
		since most tensorflow methods don't yet pass around a random state appropriately. TF random 
		states might also not play nice with tf.functions:  
			https://www.tensorflow.org/api_docs/python/tf/random/set_global_generator 
	'''
	np_random = np.random.RandomState(seed)
	tf_seed   = np_random.randint(1e10, dtype=np.int64) # tf can't take None as seed
	tf_random = tf.random.Generator.from_seed(tf_seed)
	tf.random.set_global_generator(tf_random)
	return {'np_random' : np_random, 'tf_random' : tf_random}
	

def ensure_format(arr):
	''' Ensure passed array has two dimensions [n_sample, n_feature], and add the n_feature axis if not '''
	arr = np.array(arr).copy().astype(np.float32)
	return (arr[:, None] if len(arr.shape) == 1 else arr)


def get_device(model_config):
	''' Return the tf.device a job should run on. Logic based
		on e.g. model size may be added in the future. 
	'''
	gpus = tf.config.list_physical_devices('GPU')
	cpus = tf.config.list_physical_devices('CPU')
	name = (gpus+cpus)[0].name.replace('physical_device:', '')
	return tf.device('/cpu:0')#name)