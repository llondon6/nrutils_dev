"""romSpline
=========

romSpline is an easy-to-use code for generating a reduced-order spline 
interpolant of 1d data.

romSpline uses a greedy algorithm on 1d data to find a nearly minimal subset of 
samples for recovering the original data, to the requested tolerance, with 
a univariate spline interpolant. This code should be useful for downsampling 
or compressing large data arrays to their essential components needed for 
reconstructing the original information. The degree of downsampling is 
often significant (e.g., orders of magnitude) for relatively smooth data.

Future versions of romSpline may include support for multi-dimensional data 
and additional ways for estimating the reduced-order spline interpolation error
on data in the original set being compressed.

See the accompanying IPython notebook (romSpline_example.ipynb) for a 
tutorial on using the code. See also errors_example.ipynb for a tutorial
on using the interpolation error assessment and estimation features of the 
romSpline.

If you find this code useful for your publication work then please
cite the code repository (www.bitbucket.org/chadgalley/romSpline)
and the corresponding paper that discusses and characterizes the
reduced-order spline method (available at https://arxiv.org/abs/1611.07529):

Chad R. Galley and Patricia Schmidt
"Fast and efficient evaluation of gravitational waveforms via reduced-order spline interpolation"
[arxiv: 1611.07529] (2016)

"""

__copyright__ = "Copyright (C) 2015 Chad Galley"
__author__ = "Chad Galley"
__email__ = "crgalley@tapir.caltech.edu, crgalley@gmail.com"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


############################
# Class for storing states #
#    of module imports     #
############################

class _ImportStates(object):
  """Class container for state of module imports."""
  
  def __init__(self):
    # Try importing numpy module
    try:
      import numpy as np
    except:
      # Raise an exception if numpy can't be imported
      raise Exception, "Error: Cannot import `NumPy` module."
    
    # Try importing scipy.interpolate.UnivariateSpline class
    try:
      from scipy.interpolate import UnivariateSpline
    except:
      # Raise an exception if UnivariateSpline can't be imported
      # This class is crucial to RomSpline
      raise Exception, "Error: Cannot import `scipy.interpolate.UnivariateSpline` class."
      
    # Try importing h5py module
    try:
      import h5py
      self._H5PY = True
    except:
      print "Warning: Cannot import `h5py` module. File I/O features will be limited to text formats."
      self._H5PY = False
    
    # Try importing matplotlib module
    try:
      import matplotlib.pyplot as plt
      self._MATPLOTLIB = True
    except:
      print "Warning: Cannot import `matplotlib.pyplot` module."
      self._MATPLOTLIB = False
    
    # Try importing futures module
    try:
      from concurrent.futures import ProcessPoolExecutor, wait, as_completed
      self._FUTURES = True
    except:
      print "Warning: Cannot import `futures` module. Try running `pip install futures` to install."
      self._FUTURES = False
    
    # Try importing multiprocessing module
    try:
      from multiprocessing import cpu_count
      self._MP = True
    except:
      print "Warning: Cannot import `multiprocessing` module."
      self._MP = False
    
    # If can import both futures and multiprocessing modules 
    # then can do parallelization
    # TODO: I should just use a simple Pool process so there's not
    #       an extra dependency on concurrent.futures...
    if self._FUTURES and self._MP:
      self._PARALLEL = True
    else:
      print "Warning: Parallel computation options will be unavailable."
      self._PARALLEL = False

state = _ImportStates()


#####################
# Import submodules #
#####################

from greedy import *            # For building reduced-order splines
from convergence import *       # For studying convergence
from random_seeds import *      # For studying the effect of seed points on reduced data sizes
from cross_validation import *  # For estimating (global) interpolation errors
from build_spline import *      # Convenience module for bulding reduced-order spline
                                # with a global interpolation error estimate from cross-validation
from example import *         # Built-in function for testing and demonstration purposes
from regression import *        # Regression testing

