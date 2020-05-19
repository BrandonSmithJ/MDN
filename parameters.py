from .meta import SENSOR_LABEL
import argparse


parser = argparse.ArgumentParser(epilog="""
	Passing a filename will estimate the desired parameter from the Rrs 
	contained in that file. Otherwise, a model will be trained (if not 
	already existing), and estimates will be made for the testing data.\n
""")

parser.add_argument("filename",    nargs  ="?",          help="CSV file containing Rrs values to estimate from")
parser.add_argument("--model_loc", default="Model",      help="Location of trained models")
parser.add_argument("--train_loc", default="/media/brandon/NASA/Data/Train", help="Location of training data")
parser.add_argument("--test_loc",  default="/media/brandon/NASA/Data/Test",  help="Location of testing data")
parser.add_argument("--n_redraws", default=50,     type=int,   help="Number of plot redraws during training (i.e. updates plot every n_iter / n_redraws iterations); only used with --plot_loss.")
parser.add_argument("--n_rounds",  default=10,     type=int,   help="Number of models to fit, with median output as the final estimate")


''' Flags '''
parser.add_argument("--threshold", default=None,   type=float, help="Output the maximum prior estimate when the prior is above this threshold, and the weighted average estimate otherwise. Set to None, thresholding is not used.")
parser.add_argument("--avg_est",   action ="store_true", help="Use the prior probability weighted mean as the estimate. Otherwise, use maximum prior.")
parser.add_argument("--no_save",   action ="store_true", help="Do not save the model after training")
parser.add_argument("--no_load",   action ="store_true", help="Do load a saved model (and overwrite, if not no_save)")
parser.add_argument("--verbose",   action ="store_true", help="Verbose output printing")
parser.add_argument("--silent",    action ="store_true", help="Turn off all printing")
parser.add_argument("--plot_loss", action ="store_true", help="Plot the model loss while training")
parser.add_argument("--darktheme", action ="store_true", help="Use a dark color scheme in plots")
parser.add_argument("--animate",   action ="store_true", help="Store the training progress as an animation (mp4)")
parser.add_argument("--plot_map",  action ="store_true", help="Plot a world map showing locations of the data (if available)")


''' Flags which require model retrain if changed '''
update = parser.add_argument_group('Model Parameters', 'Parameters which require a new model to be trained if they are changed')
update.add_argument("--sat_bands", action ="store_true", help="Use l2gen provided satellite bands")
update.add_argument("--benchmark", action ="store_true", help="Train only on partial dataset, and use remaining to benchmark")
update.add_argument("--product",   default="chl",        help="Product to estimate")
update.add_argument("--sensor",    default="OLI",        help="Sensor to estimate from", choices=SENSOR_LABEL)
update.add_argument("--align",     default=None,         help="Comma-separated list of sensors to align data with. Passing \"all\" uses all sensors.", choices=['all']+list(SENSOR_LABEL))
update.add_argument("--model_lbl", default="",      	 help="Label for a model")
update.add_argument("--seed",      default=1234,   type=int,   help="Random seed")


''' Flags which have a yet undecided default value '''
# update.add_argument("--no_noise",  action ="store_true", help="Do not add noise when training the model")
update.add_argument("--add_noise", action ="store_true", help="Add noise when training the model")

# update.add_argument("--no_ratio",  action ="store_true", help="Do not add band ratios as input features")
update.add_argument("--add_ratio", action ="store_true", help="Add band ratios as input features")

update.add_argument("--fix_tchl",  action ="store_true", help="Correct chl for pheopigments")
# update.add_argument("--keep_tchl", action ="store_true", help="Do not correct chl for pheopigments")

update.add_argument("--no_bagging",action ="store_true", help="Do not use bagging when training in multiple trials")
# update.add_argument("--bagging",   action ="store_true", help="Use bagging when training in multiple trials")

update.add_argument("--boosting",  action ="store_true", help="Use boosting when training in multiple trials")
# update.add_argument("--no_boosting",action ="store_true", help="Do not use boosting when training in multiple trials")

# parser.add_argument("--no_cache",  action ="store_true", help="Do not use cached training data")
parser.add_argument("--use_cache", action ="store_true", help="Use cached training data")

parser.add_argument("--use_sim", action ="store_true", help="Use simulated training data")



''' Hyperparameters '''
hypers = parser.add_argument_group('Hyperparameters', 'Hyperparameters used in training the model (also requires model retrain if changed)') 
hypers.add_argument("--n_iter",    default=100000, type=int,   help="Number of iterations to train the model")
hypers.add_argument("--n_mix",     default=5,      type=int,   help="Number of gaussians to fit in the mixture model")
hypers.add_argument("--batch",     default=128,    type=int,   help="Number of samples in a training batch")
hypers.add_argument("--n_hidden",  default=100,    type=int,   help="Number of neurons per hidden layer")
hypers.add_argument("--n_layers",  default=5,      type=int,   help="Number of hidden layers")
hypers.add_argument("--lr", 	   default=1e-3,   type=float, help="Learning rate")
hypers.add_argument("--l2", 	   default=1e-3,   type=float, help="L2 regularization")
hypers.add_argument("--epsilon",   default=1e-3,   type=float, help="Variance regularization (ensures covariance has a valid decomposition)")


testdata = parser.add_mutually_exclusive_group()
testdata.add_argument("--all_test",   action="store_const", dest="test_set", const="all")
testdata.add_argument("--paper_test", action="store_const", dest="test_set", const="paper")
parser.set_defaults(test_set='all', use_sim=False)



def get_args(kwargs={}, use_cmdline=True, **kwargs2):
	kwargs2.update(kwargs)

	if use_cmdline:	args = parser.parse_args()
	else:           args = parser.parse_args([])
	
	for k, v in kwargs2.items():
		setattr(args, k, v)
	return args