#!/usr/bin/env python

#
from distutils.core import setup
from setuptools import find_packages

#
setup(name='nrutils',
      version='1.0',
      description='Python Utilities for Numerical Reltivity Data Analysis',
      author='Lionel London',
      author_email='lionel.london@ligo.org',
      packages=find_packages(),
      include_package_data=True,
      package_dir={'nrutils': 'nrutils'},
      url='https://github.com/llondon6/nrutils',
      download_url='https://github.com/llondon6/nrutils/archive/master.zip',
      install_requires=['h5py','numpy','scipy','matplotlib'],
     )
