from __init__ import state

if state._MATPLOTLIB:
  import matplotlib.pyplot as plt

import numpy as np
import greedy, random_seeds, cross_validation



def build_spline(x, y, tol=1e-6, deg=None, rel=False, seeds=None, small=True, cv=True, K=10, parallel=True, verbose=True):
  """Build a reduced-order spline interpolant with requested tolerance and degree.
  Options include `small`, for randomly sampling the initial seed points to 
  find a smallest reduced data set, and `cv` for performing a Monte Carlo
  K-fold cross-validation in order to estimate a global interpolation error.
  
  Input
  -----
    x        -- samples
    y        -- data to be downsampled
    tol      -- L-infinity error tolerance for reduced-order spline 
                (default 1e-6)
    deg      -- degree of interpolating polynomials 
                (default None)
    rel      -- L-infinity error tolerance is relative to max abs of data?
                (default is False)
    seeds    -- seed indexes for greedy algorithm 
                (default is None (equally spaced))
    small    -- Build a "smallest" reduced-order spline?
                If so, how many sets of initial seed values
                should be used?
                (default True)
    cv       -- Perform a Monte Carlo K-fold cross-validation
                study to estimate the global interpolation error?
                If so, how many trial K-fold cross-validation 
                studies should be performed?
                (default True)
    K        -- number of partitions
                (default 10)
    parallel -- parallelize the computation?
                (default True)
    verbose  -- print progress to screen?
                (default True)
  
  Output
  ------
  A ReducedOrderSpline object with additional attributes called `median`,
  `mc_args`, and `mc_errors`. The `median` attribute is the median of all
  the MonteCarlo K-fold cross-validation errors and yields a global
  interpolation error estimate. The validation errors themselves are stored
  in the `mc_errors` attribute and their corresponding locations in the samples
  array `x` are stored in `mc_args`.
  
  Comments
  --------
  If deg is None then all polynomial degrees from 1 through 5 are sampled
  to find the smallest reduced data set sampled by the randomly selected
  sets of initial seed points.
  
  If a (integer) number is specified for the `small` option then the code
  will randomly select that number of sets of initial seed values to sample
  in order to estimate the smallest reduced data set. The larger the number
  the more seeds are sampled and the more likely the reported smallest 
  reduced data set is close to the actual smallest. However, the computational
  time will necessarily increase for larger a `small` input number.
  
  If a (integer) number is specified for the `cv` option then the code
  will execute a Monte Carlo K-fold cross-validation study with that 
  number of trials. Each trial is a single K-fold cross-validation study
  on one realization for randomly distributing the dataset into the K
  partitions. The larger the `cv` number is then the more accurate the
  resulting global interpolation error estimate becomes. However the
  computational time will necessarily increase for a larger `cv` input 
  number.
  """
  
  # Build a reduced-order spline with given specifications (deg, tol, etc.)
  if small is False:
    if verbose:
      print "\nBuilding the reduced-order spline...",
    if deg is None:
      deg = 5
    spline = greedy.ReducedOrderSpline(x, y, tol=tol, deg=deg, rel=rel, seeds=seeds, verbose=verbose)
    _small = False
  
  # Otherwise, sample the seed values to find a small reduced-order spline
  else:
    print "\nFinding a smallest reduced-order data set..."
    if small is True:  # Default number of seed sets to sample is 10
      num = 10
    else:              # Otherwise, `small` is the number of sets of seed points to sample
      assert type(small) is int, "Expecting integer value for option `small`."
      num = small
    spline = random_seeds.small_spline(x, y, num, tol=tol, deg=deg, rel=rel, parallel=parallel, verbose=verbose)
    _small = True
    
  
  if cv is not False:
    print "\nPerforming Monte Carlo K-fold cross-validation..."
    if cv is True:  # Default number of Monte Carlo K-fold cross-validation studies to perform
      num = 10
    else:
      assert type(cv) is int, "Expecting integer value for option `cv`."
      num = cv
    if _small:
      deg = spline._deg
    else:
      if deg is None:
        deg = 5
    cv_object = cross_validation.CrossValidation(tol=tol, rel=rel, deg=deg)
    cv_object.MonteCarloKfold(x, y, num, K=K, parallel=parallel, random=True, verbose=verbose)
    
    # Record all the validation errors and the samples at which they occur in `x`
    spline.mc_errors = cv_object.mc_errors
    spline.mc_args = cv_object.mc_args
    
    # The median is a useful global, approximate upper bound on the interpolation error
    spline.median = np.median( np.asarray(cv_object.errors).flatten() )  # Necessary to flatten?
  
  return spline


