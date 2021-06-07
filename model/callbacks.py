from .TrainingPlot import TrainingPlot
from ..metrics import mdsa, sspb 

from tempfile import TemporaryDirectory
from pathlib import Path

import tensorflow as tf 
import numpy as np 


class PlottingCallback(tf.keras.callbacks.Callback):
	''' Display a real-time training progress plot '''

	def __init__(self, args, data, model):
		super(PlottingCallback, self).__init__()
		self._step_count = 0
		self.args = args
		self.TP = TrainingPlot(args, model, data)
		self.TP.setup()

	def on_train_batch_end(self, batch, logs=None):
		self._step_count += 1 
		if (self._step_count % (self.args.n_iter // self.args.n_redraws)) == 0:
			self.TP.update()
	
	def on_train_end(self, *args, **kwargs):
		self.TP.finish()



class StatsCallback(tf.keras.callbacks.Callback):
	''' Save performance statistics as the model is trained '''

	def __init__(self, args, data, mdn, metrics=[mdsa, sspb], folder='Results_gpu'):
		super(StatsCallback, self).__init__()
		self._step_count = 0
		self.start_time  = time.time()
		self.args = args
		self.data = data
		self.mdn  = mdn
		self.metrics = metrics 
		self.folder  = folder

	def on_train_batch_end(self, batch, logs=None):
		if (self._step_count % (self.args.n_iter // self.args.n_redraws)) == 0:
			all_keys = sorted(self.data.keys())
			all_data = [self.data[k]['x'] for k in all_keys]
			all_sums = np.cumsum(list(map(len, [[]] + all_data[:-1])))
			all_idxs = [slice(c, len(d)+c) for c,d in zip(all_sums, all_data)]
			all_data = np.vstack(all_data)

			# Create all estimates, transform back into original units, then split back into the original datasets
			estimates = self.mdn.predict(all_data)
			estimates = {k: estimates[idxs] for k, idxs in zip(all_keys, all_idxs)}
			assert(all([estimates[k].shape == self.data[k]['y'].shape for k in all_keys])), \
				[(estimates[k].shape, self.data[k]['y'].shape) for k in all_keys]

			save_folder = Path(self.folder, self.args.config_name).resolve()
			if not save_folder.exists():
				print(f'\nSaving training results at {save_folder}\n')
				save_folder.mkdir(parents=True, exist_ok=True)

			# Save overall dataset statistics
			round_stats_file = save_folder.joinpath(f'round_{self.args.curr_round}.csv')
			if not round_stats_file.exists() or self._step_count == 0:
				with round_stats_file.open('w+') as fn:
					fn.write(','.join(['iteration','cumulative_time'] + [f'{k}_{m.__name__}' for k in all_keys for m in self.metrics]) + '\n')

			stats = [[str(m(y1, y2)) for y1,y2 in zip(self.data[k]['y'].T, estimates[k].T)] for k in all_keys for m in self.metrics]
			stats = ','.join([f'[{s}]' for s in [','.join(stat) for stat in stats]])
			with round_stats_file.open('a+') as fn:
				fn.write(f'{self._step_count},{time.time()-self.start_time},{stats}\n')

			# Save model estimates
			save_folder = save_folder.joinpath('Estimates')
			if not save_folder.exists():
				save_folder.mkdir(parents=True, exist_ok=True)

			for k in all_keys:
				filename = save_folder.joinpath(f'round_{self.args.curr_round}_{k}.csv')
				if not filename.exists():
					with filename.open('w+') as fn:
						fn.write(f'target,{list(self.data[k]["y"][:,0])}\n')

				with filename.open('a+') as fn:
					fn.write(f'{self._step_count},{list(estimates[k][:,0])}\n')
		self._step_count += 1



class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
	''' Save models during training, and load the best performing 
		on the validation set once training is completed.
		Currently untested.
	 '''

	def __init__(self, path):
		Path(path).mkdir(exist_ok=True, parents=True)
		self.tmp_folder = TemporaryDirectory(dir=path)
		self.checkpoint = Path(self.tmp_folder.name).joinpath('checkpoint')
		super(ModelCheckpoint, self).__init__(
				filepath=self.checkpoint, save_weights_only=True,
				monitor='val_MSA', mode='min', save_best_only=True) # need to add to metrics

	def on_train_end(self, *args, **kwargs):
		self.model.load_weights(self.checkpoint)
		self.tmp_folder.cleanup()



class DecayHistory(tf.keras.callbacks.Callback):
	''' Verify tf parameters are being decayed as they should;  
		call show_plot() on object once training is completed '''

	def on_train_begin(self, logs={}):
		self.lr = []
		self.wd = []

	def on_batch_end(self, batch, logs={}):
		self.lr.append(self.model.optimizer.lr)
		self.wd.append(self.model.optimizer.weight_decay)

	def show_plot(self):
		import matplotlib.pyplot as plt 
		plt.plot(self.lr, label='learning rate')
		plt.plot(self.wd, label='weight decay')
		plt.xlabel('step')
		plt.ylabel('param value')
		plt.legend()
		plt.show()