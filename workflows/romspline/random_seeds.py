from __init__ import state

if state._MATPLOTLIB:
  import matplotlib.pyplot as plt
if state._PARALLEL:
  from concurrent.futures import ProcessPoolExecutor, wait, as_completed
  from multiprocessing import cpu_count

import numpy as np
import greedy



######################################
# Random seeds for greedy algorithm  #
######################################

def _randomSeeds(x, y, seeds, tol=1e-6, rel=False, deg=5):
  """Build reduced-order splines from an arbitrary seed for the greedy algorithm.
  See RandomSeeds for further details.
  
  Input
  -----
    x     -- samples
    y     -- data to be downsampled
    seeds -- list of integers specifying the array indexes of
             the samples used to seed the greedy algorithm
    tol   -- L-infinity error tolerance for reduced-order spline 
             (default 1e-6)
    rel   -- L-infinity error tolerance is relative to max abs of data?
             (default False)
    deg   -- degree of interpolating polynomials 
             (default 5)
  """
  # Sort the seeds as a precaution
  seeds = np.sort(seeds)
  
  # Build spline
  spline = greedy.ReducedOrderSpline(x, y, tol=tol, rel=rel, deg=deg, seeds=seeds)
  
  # Return last greedy error, size, seeds, and reduced data samples
  return spline.errors[-1], spline.size, seeds, spline.X


class RandomSeeds(object):
  """A class for studying how the initial distribution of 
  seed points effects the size of the reduced data set.
  
  The default distribution of seed points depends on the
  degree (deg) of the spline polynomial used for interpolation.
  In general, one needs deg+1 points to fit a polynomial
  of degree deg. The default seed points include the first
  and last points of the data and then deg-1 points that
  are as equally spaced as possible.
  
  In this class, a set of deg+1 seed points are randomly
  selected and a reduced-order spline is generated for
  each such seed selection. The resulting sizes of the
  reduced data are recorded.
  """
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    """Create a RandomSeeds object.
    
    Input
    -----
      tol -- L-infinity error tolerance for reduced-order spline 
             (default 1e-6)
      rel -- L-infinity error tolerance is relative to max abs of data?
             (default False)
      deg -- degree of interpolating polynomials 
             (default 5)
    """
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._made = False  # Assume that random seeds study is not made yet
  
  
  def __call__(self, x, y, Nseeds, parallel=True):
    """Make reduced-order splines for Nseeds number of random sets of seed points.
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      Nseeds   -- number of random sets of seed points
      parallel -- parallelize the computation for each random seed set?
                  (default True)
    
    Attributes
    ----------
      errors -- greedy errors associated with each random set of seed points
      sizes  -- reduced data sizes from each random set of seed points
      seeds  -- the full set of random seed points
      Xs     -- reduced data from each random set of seed points
    
    Comments
    --------
    The `parallel` option also accepts positive integers indicating the
    number of processors that the user wants to use. 
    
    Parallelization uses some features from the concurrent.futures module.
    """
    return self.make(x, y, Nseeds, parallel=parallel)
  
  
  def make(self, x, y, Nseeds, parallel=True):
    """Make reduced-order splines for Nseeds number of random sets of seed points.
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      Nseeds   -- number of random sets of seed points
      parallel -- parallelize the computation for each random seed set?
                  (default True)
    
    Attributes
    ----------
      errors -- greedy errors associated with each random set of seed points
      sizes  -- reduced data sizes from each random set of seed points
      seeds  -- the full set of random seed points
      Xs     -- reduced data from each random set of seed points
    
    Comments
    --------
    The `parallel` option also accepts positive integers indicating the
    number of processors that the user wants to use. 
    
    Parallelization uses some features from the concurrent.futures module.
    """
    
    # Allocate some memory
    self.errors = np.zeros(Nseeds, dtype='double')                # Greedy errors
    self.sizes = np.zeros(Nseeds, dtype='double')                 # Sizes of each reduced data
    self.seeds = np.zeros((Nseeds, self._deg+1), dtype='double')  # All seeds
    self.Xs = []                                                  # Reduced data for each seed
    
    if (parallel is False) or (state._PARALLEL is False):
      for nn in range(Nseeds):
        seeds = np.sort( np.random.choice(range(len(x)), self._deg+1, replace=False) )
        self.errors[nn], self.sizes[nn], self.seeds[nn], Xs = _randomSeeds(x, y, seeds, tol=self._tol, rel=self._rel, deg=self._deg)
        self.Xs.append(Xs)
    elif state._PARALLEL is True:
      # Determine the number of processes to run on
      if parallel is True:
        try:
          numprocs = cpu_count()
        except NotImplementedError:
          numprocs = 1
      else:
        numprocs = parallel
      
      # Create a parallel process executor
      executor = ProcessPoolExecutor(max_workers=numprocs)
      
      # Build reduced-order splines for many random sets of seed points
      output = []
      for nn in range(Nseeds):
        seeds = np.sort( np.random.choice(range(len(x)), self._deg+1, replace=False) )
        output.append(executor.submit(_randomSeeds, x, y, seeds, tol=self._tol, rel=self._rel, deg=self._deg))
      
      # Gather the results as they complete
      #for ii, oo in enumerate(as_completed(output)):
      for ii, oo in enumerate(output):
        self.errors[ii], self.sizes[ii], self.seeds[ii], Xs = oo.result()
        self.Xs.append(Xs)
    
    self._made = True
  
  
  def plot_sizes(self, n=20, ax=None, show=True, xlabel='Size of reduced data', ylabel='Occurrence'):
    """Plot a histogram of the reduced data sizes produced by building
    reduced-order splines from random sets of seed points.
    
    Input
    -----
      n      -- number of histogram bins
                (default 20)
      ax     -- matplotlib plot/axis object
                (default None)
      show   -- display the plot?
                (default True)
      xlabel -- x-axis label 
                (default 'Size of reduced data')
      ylabel -- y-axis label
                (default 'Occurrence')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax.hist(self.sizes, n, color='k', alpha=0.33);
      
      mean = np.mean(self.sizes)
      std = np.std(self.sizes)
      
      ax.axvline(x=mean, color='k', linewidth=2)
      ax.axvline(x=mean+std, color='k', linestyle='--', linewidth=1)
      ax.axvline(x=mean-std, color='k', linestyle='--', linewidth=1)
      
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
        if ax is None:
          return fig, ax
        else:
          return ax
    
    else:
      print "No data to plot. Run make method or set _made attribute to True."
    
  
  def plot_samples(self, x, y, ax=None, show=True, xlabel='$x$', ylabel='Size of reduced data, $n$'):
    if self._made:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      # Assemble all reduced samples into a 2d array
      longest = max(map(len, self.Xs))
      Xs = max(x)*np.ones((len(self.Xs),longest), dtype='double')
      for ii, kk in enumerate(self.Xs):
        Xs[ii][:len(kk)] = kk
        
      # Plot ranges of selected samples for each iteration
      mins, maxes = [], []
      for cc in Xs.T:
        mins.append(np.min(cc))
        maxes.append(np.max(cc))
      ax.plot(mins, np.arange(len(mins)), 'k-', maxes, np.arange(len(maxes)), 'k-')
      
      # Shade between the min and max number of reduced samples
      spline_mins = greedy.UnivariateSpline(np.sort(mins), np.arange(len(mins)), s=0)
      ax.fill_between(np.sort(maxes), spline_mins(np.sort(maxes)), np.arange(len(maxes)), facecolor='k', alpha=0.33)
      
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      
      if show:
        plt.show()
      else:
        if ax is None:
          return fig, ax
        else:
          return ax
    
    else:
      print "No data to plot. Run `make` method or set _made attribute to True."



def small_spline(x, y, num, tol=1e-6, deg=None, rel=False, parallel=True, verbose=False):
  """Build reduced-order splines for a number of randomly selected
  input seed points for the purpose of finding the spline with the
  smallest such reduced data over the sampled seeds. The degree of
  the spline polynomial can also be varied.
  
  Input
  -----
    x        -- samples
    y        -- data to be downsampled
    num      -- number of sets of random initial seed points
    tol      -- L-infinity error tolerance for reduced-order spline 
                (default 1e-6)
    deg      -- degree of interpolating polynomials 
                (default None)
    rel      -- L-infinity error tolerance is relative to max abs of data?
                (default False)
    parallel -- parallelize the computation of each random seed set?
                (default True)
    verbose  -- print progress to screen?
                (default False)
  
  Output
  ------
  Reduced-order spline object with the smallest reduced data size
  for the seeds (and spline degrees) sampled.
  
  Comments
  --------
  If deg is None then polynomials of degrees 1-5 are also sampled, in
  addition to the random sets of initial seeds.
  """
  
  if deg is None:                  # Use all integers in [1,5]
    degs = range(1, 6)
  else:
    if len(np.shape(deg)) == 1:    # deg is a list or array
      degs = deg
    elif len(np.shape(deg)) == 0:  # deg is a number
      degs = [deg]
    else:
      raise Exception, "Input for `deg` option not recognized."
  for dd in degs:
    assert (dd in range(1,6)), "Expecting degree(s) to be one or more integers in [1,5]."
  
  size = len(x)+1  # Add 1 to guarantee that a smaller size will be found below
  
  for ii, dd in enumerate(degs):
    if verbose:
      print "Smallest spline for degree {} is...".format(dd),
    
    # Sample from the set of all possible initial 
    # seeds for this polynomial degree
    rand = RandomSeeds(tol=tol, deg=dd, rel=rel)
    rand.make(x, y, num, parallel=parallel)
    
    if verbose:  # Print smallest size found in sample
      print int(np.min(rand.sizes))
    
    # Find smallest spline in the sample for this polynomial degree
    imin = np.argmin(rand.sizes)
    if rand.sizes[imin] < size:
      size = rand.sizes[imin]
      seed = rand.seeds[imin]
      degree = dd
  
  # Output the reduced-order spline object for the smallest case
  return greedy.ReducedOrderSpline(x, y, tol=tol, deg=degree, rel=rel, seeds=seed)


