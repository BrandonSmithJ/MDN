import argparse

parser = argparse.ArgumentParser(epilog="""
	Passing a filename will estimate the desired parameter from the Rrs 
	contained in that file. Otherwise, a model will be trained (if not 
	already existing), and estimates will be made for the testing data.\n
""")

parser.add_argument("filename",    nargs  ="?",          help="CSV file containing Rrs values to estimate from")
parser.add_argument("--model_loc", default="Weights",    help="Location of trained model weights")
# parser.add_argument("--data_loc",  default="/media/brandon/NASA/Data/Insitu",  help="Location of in situ data")
# parser.add_argument("--sim_loc",   default="/media/brandon/NASA/Data/Simulated", help="Location of simulated data")
parser.add_argument("--data_loc",  default="D:/Data/Insitu",    help="Location of in situ data")
parser.add_argument("--sim_loc",   default="D:/Data/Simulated", help="Location of simulated data")
parser.add_argument("--n_redraws", default=50,     type=int,   help="Number of plot redraws during training (i.e. updates plot every n_iter / n_redraws iterations); only used with --plot_loss.")
parser.add_argument("--n_rounds",  default=10,     type=int,   help="Number of models to fit, with median output as the final estimate")


''' Flags '''
parser.add_argument("--threshold", default=None,   type=float, help="Output the maximum prior estimate when the prior is above this threshold, and the weighted average estimate otherwise. Set to None, thresholding is not used")
parser.add_argument("--avg_est",   action ="store_true", help="Use the prior probability weighted mean as the estimate; otherwise, use maximum prior")
parser.add_argument("--no_save",   action ="store_true", help="Do not save the model after training")
parser.add_argument("--no_load",   action ="store_true", help="Do load a saved model (and overwrite, if not no_save)")
parser.add_argument("--verbose",   action ="store_true", help="Verbose output printing")
parser.add_argument("--silent",    action ="store_true", help="Turn off all printing")
parser.add_argument("--plot_loss", action ="store_true", help="Plot the model loss while training")
parser.add_argument("--darktheme", action ="store_true", help="Use a dark color scheme in plots")
parser.add_argument("--animate",   action ="store_true", help="Store the training progress as an animation (mp4)")
parser.add_argument("--save_data", action ="store_true", help="Save the data used for the given args")
parser.add_argument("--save_stats",action ="store_true", help="Store partial training statistics & estimates for later analysis")
parser.add_argument("--LOO_CV",    action ="store_true", help="Leave-one-out cross validation")


''' Flags which require model retrain if changed '''
update = parser.add_argument_group('Model Parameters', 'Parameters which require a new model to be trained if they are changed')
update.add_argument("--sat_bands", action ="store_true", help="Use bands specific to certain products when utilizing satellite retrieved spectra")
update.add_argument("--benchmark", action ="store_true", help="Train only on partial dataset, and use remaining to benchmark")
update.add_argument("--product",   default="chl",        help="Product to estimate")
update.add_argument("--sensor",    default="OLI",        help="Sensor to estimate from (See meta.py for available options)")
update.add_argument("--align",     default=None,         help="Comma-separated list of sensors to align data with; passing \"all\" uses all sensors (See meta.py for available options)")
update.add_argument("--model_lbl", default="",      	 help="Label for a model")
update.add_argument("--seed",      default=42, type=int, help="Random seed")
update.add_argument("--subset",    default='',           help="Comma-separated list of datasets to use when fetching data")


''' Flags which have a yet undecided default value '''
flags = parser.add_argument_group('Model Flags', 'Flags which require a new model to be trained if they are changed')

# flags.add_argument("--no_noise",  action ="store_true", help="Do not add noise when training the model")
flags.add_argument("--use_noise",     action ="store_true", help="Add noise when training the model")

# flags.add_argument("--no_ratio",  action ="store_true", help="Do not add band ratios as input features")
flags.add_argument("--use_ratio",     action ="store_true", help="Add band ratios as input features")

flags.add_argument("--use_auc",       action ="store_true", help="Normalize input features using AUC")

flags.add_argument("--use_tchlfix",   action ="store_true", help="Correct chl for pheopigments")
# flags.add_argument("--no_tchlfix", action ="store_true", help="Do not correct chl for pheopigments")

# flags.add_argument("--no_cache",  action ="store_true", help="Do not use any cached data")
# flags.add_argument("--use_cache", action ="store_true", help="Use cached data, if available")

flags.add_argument("--use_boosting",  action ="store_true", help="Use boosting when training in multiple trials (no longer implemented)")
# flags.add_argument("--no_boosting",action ="store_true", help="Do not use boosting when training in multiple trials")

flags.add_argument("--no_bagging",    action ="store_true", help="Do not use bagging when training in multiple trials")
# flags.add_argument("--use_bagging",   action ="store_true", help="Use bagging when training in multiple trials")

flags.add_argument("--use_sim",       action ="store_true", help="Use simulated training data")

flags.add_argument("--use_excl_Rrs",  action ="store_true", help="Drop raw Rrs features from the input when using ratio features")
flags.add_argument("--use_all_ratio", action ="store_true", help="Use exhaustive list of ratio features instead of only those found in literature (should be combined with --use_kbest)")

flags.add_argument("--use_kbest", type=int, nargs='?', const=5, default=0, help="Select top K features to use as input, based on mutual information")


''' Hyperparameters '''
hypers = parser.add_argument_group('Hyperparameters', 'Hyperparameters used in training the model (also requires model retrain if changed)') 
hypers.add_argument("--n_iter",      default=10000,  type=int,   help="Number of iterations to train the model")
hypers.add_argument("--n_mix",       default=5,      type=int,   help="Number of gaussians to fit in the mixture model")
hypers.add_argument("--batch",       default=128,    type=int,   help="Number of samples in a training batch")
hypers.add_argument("--n_hidden",    default=100,    type=int,   help="Number of neurons per hidden layer")
hypers.add_argument("--n_layers",    default=5,      type=int,   help="Number of hidden layers")
hypers.add_argument("--imputations", default=5,      type=int,   help="Number of samples used for imputation when handling NaNs in the target")
hypers.add_argument("--lr", 	     default=1e-3,   type=float, help="Learning rate")
hypers.add_argument("--l2", 	     default=1e-3,   type=float, help="L2 regularization")
hypers.add_argument("--epsilon",     default=1e-3,   type=float, help="Variance regularization (ensures covariance has a valid decomposition)")


dataset = parser.add_mutually_exclusive_group()
dataset.add_argument("--all_test",       action="store_const", dest="dataset", const="all")
dataset.add_argument("--sentinel_paper", action="store_const", dest="dataset", const="sentinel_paper")
parser.set_defaults(dataset='all', use_sim=False)



def get_args(kwargs={}, use_cmdline=True, **kwargs2):
	kwargs2.update(kwargs)

	if use_cmdline:	args = parser.parse_args()
	else:           args = parser.parse_args([])
	
	for k, v in kwargs2.items():
		setattr(args, k, v)
	return args