from setuptools import setup
from __version__ import __version__

setup(
    name='MDN',
    version=__version__,
    description='Mixture Density Network',
    author='Brandon Smith',
    author_email='b.smith@nasa.gov',
    url='https://github.com/BrandonSmithJ/MDN',
    package_dir={'MDN': ''},
    packages=['MDN'],
    include_package_data=True,
)
