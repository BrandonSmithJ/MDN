from pathlib import Path 
from importlib import import_module
from collections import defaultdict as dd
import numpy as np 
import pkgutil, traceback

from .. import image_estimates, get_tile_data, get_sensor_bands 
from ..meta import SENSOR_BANDS
from ..Benchmarks.utils import get_benchmark_models
from ..benchmarks import get_methods


SHOW_TRACEBACK = True # Show the stack trace for any exceptions 


def _test_success(f, *args, _tab_level=2, **kwargs):
	''' Determine whether the given function throws an exception '''
	output = 'PASS'
	try: f(*args, **kwargs)
	except Exception as e:
		output = 'FAIL\n' + ''.join(['\t']*_tab_level) + f'Exception: {e}'
		if SHOW_TRACEBACK: output += f'\n{traceback.format_exc()}\n'
	return output


def _get_sensors(model_dir='Model'):
	''' Get all sensor folders within the Model directory '''
	model_dir = Path(__file__).parent.parent.joinpath(model_dir)
	sensors   = [path.stem for path in model_dir.glob('*') if path.stem in SENSOR_BANDS]
	assert(len(sensors)), f'No valid sensors found in directory "{model_dir}"; valid sensors:\n{SENSOR_BANDS.keys()}'
	return sorted(sensors)


def _get_benchmarks(benchmark_dir='Benchmarks'):
	''' Get all benchmark models within the Benchmarks directory '''
	bench_dir  = Path(__file__).parent.parent.joinpath(benchmark_dir).resolve()
	products   = [path.stem for path in bench_dir.glob('*') if path.is_dir() and path.stem[0] != '_']
	benchmarks = {product: get_benchmark_models(product) for product in products}
	assert(len(benchmarks)), f'No benchmarks found in directory "{bench_dir}"'
	return benchmarks


def test_image_estimates():
	''' Test image_estimates() with random data '''

	def test_sensor(sensor):
		rand_data = [np.random.rand(3,3) for band in get_sensor_bands(sensor)]
		estimates = image_estimates(rand_data, sensor=sensor, silent=True)
		chlor_a   = estimates[0]
		assert(chlor_a.shape == (3,3)), f'Unexpected shape found: {chlor_a.shape}'

	sensors = _get_sensors()
	longest = max(map(len, sensors))

	for sensor in sensors:
		print(f'\t{sensor:>{longest}}: { _test_success(test_sensor, sensor) }')


def test_benchmarks():
	''' Test all benchmarks '''

	def test_benchmark(model, sensor='OLCI'):
		wavelengths = np.arange(350, 2500)
		sample_Rrs  = np.ones((1, len(wavelengths)))
		model(sample_Rrs, wavelengths, sensor)
	
	product_benchmarks = _get_benchmarks()
	longest_product    = max(map(len, product_benchmarks.keys()))

	for product, benchmarks in product_benchmarks.items():
		print(f'\n\t{product:>{longest_product}}: {len(benchmarks)} benchmarks found')

		longest = max(map(len, benchmarks))
		for name, model in benchmarks.items():
			print(f'\t\t\t{name:>{longest}}: { _test_success(test_benchmark, model, _tab_level=4) }')



if __name__ == '__main__':	
	test_functions = [test_benchmarks, test_image_estimates]

	for function in test_functions:
		print(f'\n{function.__name__}')

		try:                   function()
		except Exception as e: print(f'\tFailed to run {function.__name__}: {e}')
			
