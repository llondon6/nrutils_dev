from __init__ import state

if state._MATPLOTLIB:
  import matplotlib.pyplot as plt
if state._PARALLEL:
  from __init__ import ProcessPoolExecutor, wait, as_completed, cpu_count

import numpy as np
import greedy
from helpers import *



###########################
# K-fold cross validation #
###########################

def _cross_validation(x, y, validation, training, tol=1e-6, rel=False, deg=5):
  """K-fold cross validation on a given partition
  
  Input
  -----
    x          -- samples
    y          -- data values at samples
    validation -- validation subset of data indexes
    training   -- training subset of data indexes
    tol        -- L-infinity error tolerance for reduced-order spline 
                  (default 1e-6)
    rel        -- L-infinity error tolerance is relative to max abs of data?
                  (default False)
    deg        -- degree of interpolating polynomials 
                  (default 5)
  
  Output
  ------
    sample index where largest validation error occurs
    largest validation error
  
  Comment
  -------
  The union of the validation and training subsets must contain all the
  data array indexes.
  """
  
  assert len(x) == len(y), "Expecting x and y arrays to have same lengths."
  assert len(np.hstack([validation, training])) == len(x), "Expecting union of validation and training sets to have same lengths as x and y arrays."
  
  # Form trial data
  x_training = x[training]
  y_training = y[training]
  
  # Build trial spline
  spline = greedy._greedy(x_training, y_training, tol=tol, rel=rel, deg=deg)[0]
  
  # Compute L-infinity errors between training and data in validation partition
  errors = np.abs(spline(x[validation]) - y[validation])
  
  return errors, np.max(errors), validation[np.argmax(errors)]


class CrossValidation(object):
  """A class for performing K-fold cross-validation studies to assess
  the errors in reduced-order spline interpolation.
  
  A K-fold cross-validation study consists of randomly distributing
  the original dataset into K partitions such that their union, when
  sorted, is the original data. Select the first partition for validation
  and build a reduced-order spline interpolant on the remaining K-1 
  partitions. Compute the L-infinty error between the resulting trial
  spline and the actual data in the validation partition. Repeat using
  each of the K partitions as a validation set. This results in K
  validation errors, one for each validation partition. The mean (median)
  of these is the mean (median) validation error. K-fold cross-validation
  is implemented in the `Kfold` method.
  
  This study can be repeated for different realizations of the random
  distribution of the original data into the K partitions. Each draw is from
  an independent and identical distribution so standard statistics applies
  to the mean (median) validation errors computed for the ensemble of all
  draws. Use the `MonteCarloKfold` method for this kind of study.
  """
  
  def __init__(self, tol=1e-6, rel=False, deg=5):
    """Create a CrossValidation object.
    
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
    self._made = False
  
  
  def Kfold(self, x, y, K=10, parallel=True, random=True):
    """K-fold cross-validation
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      K        -- number of partitions
                  (default 10)
      parallel -- parallelize the computation over each partition?
                  (default True)
      random   -- fill the partitions randomly from the data?
                  (default True)
    
    Attributes
    ----------
      args   -- data array indexes of largest validation errors 
                on each validation set
      errors -- validation errors on each validation set
    
    Comments
    --------
    The default option for `K` is 10 so that a random 10% subset of the 
    data is used to validate the trial reduced-order splines.
    
    If `random` option is False then the partitions are filled with the
    data in sequential order. This is not a good idea for trying to 
    assess interpolation errors because interpolating over large regions
    of data will likely incur large errors. This option may be removed in 
    future romSpline releases.
    """
    
    assert len(x) == len(y), "Expecting input data to have same length."
    
    self._K = int(K)
    self._size = len(x)
    
    # Allocate some memory
    self.args = np.zeros(K, dtype='int')
    self.errors = np.zeros(K, dtype='double')
    
    # Divide into K nearly equal partitions
    if random:
      self._partitions = random_partitions(self._size, self._K)
    else:
      self._partitions = partitions(self._size, self._K)
    
    if (parallel is False) or (state._PARALLEL is False):
      for ii in range(self._K):
        validation, training = partitions_to_sets(ii, self._partitions, sort=True)
        errors, self.errors[ii], self.args[ii] = _cross_validation(x, y, validation, training, tol=self._tol, rel=self._rel, deg=self._deg)
    
    # TODO: Change parallelization to simple Pool processes
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
      
      # Compute the spline errors on each validation partition
      output = []
      for ii in range(self._K):
        validation, training = partitions_to_sets(ii, self._partitions, sort=True)
        output.append( executor.submit(_cross_validation, x, y, validation, training, tol=self._tol, rel=self._rel, deg=self._deg) )
      
      # Gather the results
      for ii, ee in enumerate(output):
        errors, self.errors[ii], self.args[ii] = ee.result()
    
    mask = (self.errors >=self._tol)
    self.args_ge_tol = self.args[mask]
    self.errors_ge_tol = self.errors[mask]
    
    self._made = True
  
  
  def MonteCarloKfold(self, x, y, n, K=10, parallel=True, random=True, verbose=False):
    """Perform a number n of K-fold cross-validations
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      n        -- number of times to repeat K-fold cross-validation
                  with a random distribution of the data into the 
                  K partitions
      K        -- number of partitions
                  (default 10)
      parallel -- parallelize the computation over each partition?
                  (default True)
      random   -- fill the partitions randomly from the data?
                  (default True)
      verbose  -- print progress to screen?
                  (default False)
    
    Attributes
    ----------
      all_args   -- array of data array indexes of validation errors on each
                   validation set
      all_errors -- array of validation errors on each realized validation set
    
    Comments
    --------
    The default option for `K` is 10 so that a random 10% subset of the 
    data is used to validate the trial reduced-order splines.
    
    If `random` option is False then the partitions are filled with the
    data in sequential order. This is not a good idea for trying to 
    assess interpolation errors because interpolating over large regions
    of data will likely incur large errors. This option may be removed in 
    future romSpline releases.
    """
    
    # Allocate memory
    self.all_args = np.zeros((n,K), dtype='int')
    self.all_errors = np.zeros((n,K), dtype='double')
    self.all_args_ge_tol = []
    self.all_errors_ge_tol = []
    for nn in range(n):
      
      if verbose and not (nn+1)%10:
        print "Trials completed:", nn+1
      
      self.Kfold(x, y, K=K, parallel=parallel, random=random)
      self.all_args[nn] = self.args
      self.all_errors[nn] = self.errors
      
      self.all_args_ge_tol.append( list(self.args[self.errors >= self._tol]) )
      self.all_errors_ge_tol.append( list(self.errors[self.errors >= self._tol]) )
  
  
  def LeaveOneOut(self, x, y, parallel=True):
    """Leave-one-out cross-validation
    
    Input
    -----
      x        -- samples
      y        -- data to be downsampled
      parallel -- parallelize the computation over each partition?
                  (default True)
      
    Attributes
    ----------
      args   -- data array indexes of largest validation errors 
                on each validation set
      errors -- validation errors on each validation set
    
    Comments
    --------
    Leave-one-out is the same as K-fold cross-validation where K
    equals the number of samples.
    """
    return self.Kfold(x, y, K=len(x), parallel=parallel, random=False)
  
  
  def stats(self):
    """Compute means and medians of each set of validation errors from 
    a Monte Carlo K-fold cross-validation study. The validation errors
    considered are those greater than or equal to the reduced-order
    spline tolerance.
    
    Attributes Made
    ---------------
      means_all_errors_ge_tol   -- means of each set of validation errors
                                   that are greater than or equal to the
                                   reduced-order spline tolerance
      medians_all_errors_ge_tol -- medians of each set of validation errors
                                   that are greater than or equal to the
                                   reduced-order spline tolerance
    
    Comments
    --------
    A single trial of a Monte Carlo K-fold cross-validation study produces
    K validation errors, which are the largest absolute errors in each 
    of the K subsets. Of these K validation errors, only those greater than
    or equal to the tolerance used to build the reduced-order spline are
    used to compute the means and medians here. Each mean and median is
    computed for each of the Monte Carlo trials so that 100 trials yields
    100 such means and medians.
    """
    
    # Mean of error means and sample standard deviation
    _means = []
    _medians = []
    for ee in self.all_errors_ge_tol:
      if ee != []:
        _means.append( np.mean(ee) )
        _medians.append( np.median(ee) )
    
    self.means_all_errors_ge_tol = _means
    self.medians_all_errors_ge_tol = _medians
    
    # Specific quantiles of error means
    #self.mc_lower_quantile, self.mc_median, self.mc_upper_quantile = mquantiles(self.mc_mean_errors, prob=[lq, 0.5, uq])
  
  
  def plot_partition_errors(self, ax=None, show=True, color='k', marker='o', linestyle='-'):
    """Plot the largest interpolation error for each validation
    partition from a K-fold cross-validation study
    
    Input
    -----
      ax        -- matplotlib plot/axis object
                   (default None)
      show      -- display the plot?
                   (default True)
      color     -- data color
                   (default 'k')
      marker    -- data marker style
                   (default 'o')
      linestyle -- data line style
                   (default '-')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    if self._made and state._MATPLOTLIB:
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      x_plot = range(len(self._partitions))
      ax.plot(x_plot, self.errors, color=color, marker=marker, linestyle=linestyle)
      
      ax.set_xlabel('Partition')
      ax.set_ylabel('Validation error')
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data attributes to plot."
  
  
  def plot_monte_carlo_errors(self, x=None, n=20, axes='plot', ax=None, show=True, color='k', marker='.'):
    """Plot all the Monte Carlo K-fold cross-validation errors
    versus the samples at which the error is recorded 
    in each of the validation partitions.
    
    Input
    -----
      x      -- samples
                (default None)
      n      -- number of bins if plotting a histogram
      axes   -- axis scales for plotting
                (default 'plot')
      ax     -- matplotlib plot/axis object
                (default None)
      show   -- display the plot?
                (default True)
      color  -- data color
                (default 'k')
      marker -- data marker style
                (default '.')
    """
    
    if self._made and state._MATPLOTLIB:
      
      self.stats()
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      # Flatten args and errors arrays
      # generated from MonteCarloKFold method
      args = np.asarray(self.all_args).flatten()
      errors = np.asarray(self.all_errors).flatten()
      
      # Set up x-data if not plotting a histogram
      if axes != 'hist':
        ax.set_ylabel('Validation errors')
        if x is None:
          xplot = args
          ax.set_xlabel('Data array indexes')
        else:
          xplot = x[args]
          ax.set_xlabel('$x$')
      
      if axes == 'semilogy':
        ax.semilogy(xplot, errors, color=color, marker=marker, linestyle='')
      elif axes == 'plot':
        ax.plot(xplot, errors, color=color, marker=marker, linestyle='')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
      elif axes == 'hist':
        ax.hist(np.log10(errors), n, alpha=0.3, color=color)
        ax.set_xlabel('$\\log_{10}$(Validation errors)')
        ax.set_ylabel('Occurrence')
      
      if show:  # Display the plot
        plt.show()
      else:     # Otherwise, return plot objects for editing the plot in the future
        if ax is None:
          return fig, ax
        else:
          return ax
      
    else:
      print "No data attributes to plot."


