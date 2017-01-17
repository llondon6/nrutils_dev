from __init__ import _ImportStates

state = _ImportStates()
if state._MATPLOTLIB:
  import matplotlib.pyplot as plt
if state._H5PY:
  import h5py

import numpy as np, os
from scipy.interpolate import UnivariateSpline



########################
# Class for building a #
# reduced-order spline #
########################

class ReducedOrderSpline(object):
  """Class for building a reduced-order spline interpolant of one-dimensional data"""
  
  def __init__(self, x=None, y=None, deg=5, tol=1e-6, rel=False, verbose=False, seeds=None):
    self._deg = deg
    self._rel = rel
    self._made = False
  
    # Run greedy algorithm if data is given for class instantiation
    if y is not None:
      if x is None:
        x = np.arange(len(y))
      assert len(x) == len(y), "Array sizes must be equal."
      self.greedy(x, y, deg=deg, tol=tol, rel=rel, verbose=verbose, seeds=seeds)
  
  
  def seed(self, x, seeds=None):
    """Seed the greedy algorithm with (deg+1) evenly spaced indices"""
    if seeds is None:
      f = lambda m, n: [ii*n//m + n//(2*m) for ii in range(m)]
      self.indices = np.sort(np.hstack([[0, len(x)-1], f(self._deg-1, len(x))]))
    else:
      self.indices = seeds
    self.errors = []
    
    # Keep track of unsorted indices selected by greedy algorithm
    self.args = self.indices[:]
  
  
  def greedy(self, x, y, tol=1e-6, rel=False, deg=5, verbose=False, seeds=None):
    """
    Greedy algorithm for finding optimal knots to build a reduced-order 
    spline interpolant
    
    Input
    -----
    x       -- samples
    y       -- data to be downsampled
    deg     -- degree of interpolating polynomials 
               (default 5)
    tol     -- L-infinity error tolerance for reduced-order spline 
               (default 1e-6)
    rel     -- L-infinity error tolerance is relative to max abs of data?
               (default is False)
    verbose -- write greedy algorithm output to screen 
               (default False)
    seeds   -- seed indexes for greedy algorithm 
               (default is None (equally spaced))
    
    Output attributes
    -----------------
    errors      -- L-infinity errors of successive spline interpolants 
                   produced by greedy algorithm until tol is reached
    X           -- Nearly optimal data samples for spline interpolant of given order
    Y           -- Nearly optimal data values for spline interpolant of given order
    indices     -- Sorted array indices of greedy-selected data
    args        -- Unsorted array indices of greedy-selected data
    size        -- Number of points retained for reduced-order spline
    compression -- Ratio of original data length to size
    
    Comments
    --------
    If x is not given then interpolation is built using y array index labels
    as the samples.
    """
    
    self._deg = deg
    self._rel = rel
    
    if rel:
      ymax = np.max(np.abs(y))
      assert ymax > 0., "All data samples are zero."
    else:
      ymax = 1.
    self._tol = tol
    #self.tol = self._tol*ymax
    
    # Seed greedy algorithm
    if seeds is None:
      self._seed = _seed(x, deg=self._deg, seeds=None)[0]
    else:
      self._seed = seeds
      
    # Greedy algorithm
    self._spline, self.indices, self.args, self.errors, self.tol = _greedy(x, y, tol=self._tol, rel=self._rel, deg=self._deg, verbose=verbose, seeds=seeds)
    
    # Define some attributes for spline evaluation
    self.X = x[self.indices]
    self.Y = y[self.indices]
    self.size = len(self.indices)
    self.compression = float(len(y))/self.size
    self._made = True
  
  
  def __call__(self, x, dx=0):
    """Evaluate reduced-order spline or its dx derviatives at x
    
    Input
    -----
      x  -- samples
      dx -- number of derivatives of the spline evaluation
            to compute
            (default 0)
    
    Output
    ------
    Array of reduced-order spline evaluations at the samples `x` 
    and `dx` number of derivatives computed.
    
    Comments
    --------
    The max number of derivatives that can be computed equals the
    degree of the spline.
    """
    return self.eval(x, dx=dx)
  
  
  def eval(self, x, dx=0):
    """Evaluate reduced-order spline or its dx derviatives at x
    
    Input
    -----
      x  -- samples
      dx -- number of derivatives of the spline evaluation
            to compute
            (default 0)
    
    Output
    ------
    Array of reduced-order spline evaluations at the samples `x` 
    and `dx` number of derivatives computed.
    
    Comments
    --------
    The max number of derivatives that can be computed equals the
    degree of the spline.
    """
    return self._spline(x, dx)
  
  
  def test(self, x, y, abs=True, dx=0, verbose=False):
    """Test (or verify) the reduced-order spline on some given data.
    If the data is the training set then this function verifies that
    the spline meets the requested tolerance.
    
    Input
    -----
      x       -- samples
      y       -- data values at samples
      abs     -- output absolute values of error?
                 (default True)
      dx      -- number of derivatives of the spline evaluation
                 to compute
                 (default 0)
      verbose -- print to screen the results of the test?
                 (default False)
    
    Output
    ------
    Absolute errors between the data values (y) and the reduced-
    order spline interpolant's prediction.
    
    Comments
    --------
    The max number of derivatives that can be computed equals the
    degree of the spline.
    """
    
    if self._made:
      errors = y - self._spline(x, dx)
      if abs:
        errors = np.abs(errors)
    else:
      raise Exception, "No spline interpolant to compare against. Run the greedy method."
    
    if verbose:
      print "Requested tolerance of {} met: ".format(self._tol), np.all(np.abs(errors) <= self.tol)
    
    return errors
  
  
  def plot_greedy_errors(self, rel=False, ax=None, show=True, axes='semilogy', xlabel='Size of reduced data', ylabel='Greedy errors'):
    """Plot the greedy errors versus size of the reduced data.
    
    Input
    -----
      ax     -- matplotlib plot/axis object
                (default None)
      show   -- display the plot?
                (default True)
      axes   -- axis scales for plotting
                (default 'semilogy')
      xlabel -- label of x-axis
                (default 'Size of reduced data')
      ylabel -- label of y-axis
                (default 'Greedy errors')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
    if state._MATPLOTLIB:
      
      if self._made:  # Check if spline data made
        if ax is None:
          fig, ax = plt.subplots(nrows=1, ncols=1)
        
        data = self.errors
        if rel:
          data *= self._tol / self.tol
        
        # Select the axes upon which to plot the greedy errors
        if axes == 'semilogy':
          ax.semilogy(np.arange(1, len(data)+1), data, 'k-')
        elif axes == 'semilogx':
          ax.semilogx(np.arange(1, len(data)+1), data, 'k-')
        elif axes == 'loglog':
          ax.loglog(np.arange(1, len(data)+1), data, 'k-')
        elif axes == 'plot':
          ax.plot(np.arange(1, len(data)+1), data, 'k-')
        
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
        print "No data to plot. Run `greedy` method."
    
    else:
      print "Error: matplotlib.pyplot module not imported."
  
  
  def write(self, file, slim=False):
    """
    Write spline interpolant data in HDF5 or text format
    
    Input
    =====
    file -- write data to this file 
    slim -- export minimal data to this file (i.e., exclude writing errors array)
            (default False)
    
    Comments
    ========
    Data written to file includes:
      
      deg    -- degree of spline polynomial interpolants
      tol    -- greedy algorithm threshold tolerance
      X      -- reduced data samples
      Y      -- reduced data values
      errors -- L-infinity norm of differences between 
                reduced order spline and full data set
    
    Input file can be an HDF5 file or group descriptor. Also,
    file can be a string and is expected to be in the form of
      
      /my/directory/filename.extension
    
    Currently, HDF5 and text formats are supported. If text 
    format, a directory is created with text files containing 
    the reduced order spline data.
    """
    
    if not state._H5PY:
      print "h5py module not imported. Try writing data to text (.txt) format."
      return
    
    # If file is an HDF5 file or group descriptor...
    if file.__class__ in [h5py._hl.files.File, h5py._hl.group.Group]:
      self._write(file, slim=slim)
    
    # If file is a file name...
    elif type(file) is str:
      # Get file name and extension
      types = ['.txt', '.h5', '.hdf5']
      filename, file_extension = os.path.splitext(file)
      assert file_extension in types, "File type must have extension txt, h5, or hdf5."
      
      # HDF5 format
      if file_extension == '.h5' or file_extension == '.hdf5':
        if state._H5PY:
          try:
            fp = h5py.File(file, 'w')
            isopen = True
          except:
            raise Exception, "Could not open file for writing."
          if isopen:
            self._write(fp, slim=slim)
            fp.close()
        else:
          print "Error: h5py module is not imported. Try writing data to text (.txt) format."
          return
      
      # Text format
      if file_extension == '.txt':
        # Make directory with given filename
        if not os.path.exists(file):
            os.makedirs(file)
        
        # Write polynomial degree of reduced-order spline
        fp = open(file+'/deg.txt', 'w')
        fp.write(str(self._deg))
        fp.close()
        
        # Write greedy algorithm tolerance
        fp = open(file+'/tol.txt', 'w')
        fp.write(str(self.tol))
        fp.close()
        
        # Write nearly optimal subset of x data (i.e., "knots")
        fp = open(file+'/X.txt', 'w')
        for xx in self.X:
          fp.write(str(xx)+'\n')
        fp.close()
        
        # Write nearly optimal subset of y data
        fp = open(file+'/Y.txt', 'w')
        for yy in self.Y:
          fp.write(str(yy)+'\n')
        fp.close()
        
        # Write L-infinity spline errors from greedy algorithm
        if not slim:
          fp = open(file+'/errors.txt', 'w')
          for ee in self.errors:
            fp.write(str(ee)+'\n')
          fp.close()
  
  
  def _write(self, descriptor, slim=False):
    """Write reduced order spline data to HDF5 file given a file or group descriptor"""
    if descriptor.__class__ in [h5py._hl.files.File, h5py._hl.group.Group]:
      descriptor.create_dataset('deg', data=self._deg, dtype='int')
      descriptor.create_dataset('tol', data=self.tol, dtype='double')
      descriptor.create_dataset('X', data=self.X, dtype='double', compression='gzip', shuffle=True)
      descriptor.create_dataset('Y', data=self.Y, dtype='double', compression='gzip', shuffle=True)
      if not slim:
        descriptor.create_dataset('errors', data=self.errors, dtype='double', compression='gzip', shuffle=True)
    else:
      raise Exception, "Descriptor not recognized."
  
  
  def read(self, file, group=None):
    """
    Load spline interpolant data from HDF5 or text format

    Input
    =====
    file -- load data from this file assuming form of
            /my/directory/filename.extension
    
    Valid extensions are 'h5', 'hdf5', and 'txt'.
    """
    ans = readSpline(file, group=group)
    self._spline = ans._spline
    self.X = ans.X
    self.Y = ans.Y
    self._deg = ans._deg
    self.errors = ans.errors
    self.tol = ans.tol
    self._made = True


def readSpline(file, group=None):
  """
  Load spline interpolant data from HDF5 or text format 
  without having to create a ReducedOrderSpline object.
  
  Input
  =====
  file -- load data from this file assuming form of
          /my/directory/filename.extension
  
  Output
  ======
  spline -- object that has the attributes and methods
            of ReducedOrderSpline class
  
  Valid file extensions are 'h5', 'hdf5', and 'txt'.
  """
  
  # Get file name and extension
  types = ['.txt', '.h5', '.hdf5']
  filename, file_extension = os.path.splitext(file)
  assert file_extension in types, "File type must be have extension txt, h5, or hdf5."
  
  # HDF5 format
  if file_extension == '.h5' or file_extension == '.hdf5':
    if state._H5PY:
      try:
        fp = h5py.File(file, 'r')
        isopen = True
      except:
        raise Exception, "Could not open file for reading."
      if isopen:
        gp = fp[group] if group else fp
        deg = gp['deg'][()]
        tol = gp['tol'][()]
        X = gp['X'][:]
        Y = gp['Y'][:]
        if hasattr(gp, 'errors') or 'errors' in gp.keys():
          errors = gp['errors'][:]
        else:
          errors = []
        fp.close()
        _made = True
    else:
      print "Error: h5py module is not imported."
      return
  
  # Text format
  if file_extension == '.txt':
    try:
      fp_deg = open(file+'/deg.txt', 'r')
      fp_tol = open(file+'/tol.txt', 'r')
      fp_X = open(file+'/X.txt', 'r')
      fp_Y = open(file+'/Y.txt', 'r')
      try:
        fp_errs = open(file+'/errors.txt', 'r')
        errs_isopen = True
      except:
        errs_isopen = False
      isopen = True
    except:
      raise IOError, "Could not open file(s) for reading."
    
    if isopen:
      deg = int(fp_deg.read())
      fp_deg.close()
      
      tol = float(fp_tol.read())
      fp_tol.close()
      
      X = []
      for line in fp_X:
        X.append( float(line) )
      X = np.array(X)
      fp_X.close()
      
      Y = []
      for line in fp_Y:
        Y.append( float(line) )
      Y = np.array(Y)
      fp_Y.close()
      
      errors = []
      if errs_isopen:
        for line in fp_errs:
          errors.append( float(line) )
        errors = np.array(errors)
        fp_errs.close()
      
      _made = True
  
  if _made:
    spline = ReducedOrderSpline()
    spline._spline = UnivariateSpline(X, Y, k=deg, s=0)
    spline.X = X
    spline.Y = Y
    spline._deg = deg
    spline.errors = errors
    spline.tol = tol
    spline.size = len(X)
    spline.compression = None  # Original data set not available so setting to None
    spline._made = _made
    return spline
  else:
    raise Exception, "Reduced-order spline interpolant could not be constructed from file."



#################################
# Functions for parallelization #
#################################

def _seed(x, deg=5, seeds=None):
  """Seed the greedy algorithm with (deg+1) evenly spaced indices
  
  Input
  -----
    x     -- samples
    deg   -- degree of spline polynomial
             (default 5)
    seeds -- list or array of indices of initial seed samples
             (default None)
  
  Comment
  -------
  If `seeds` is None then the code chooses the seed points to be the
  first and last indices and (deg-1) points in between that are
  as nearly equally spaced as possible.
  
  If a list or array of `seeds` are entered then these are used
  instead.
  """
  if seeds is None:
    # TODO: Check that these are actually equally spaced...
    f = lambda m, n: [ii*n//m + n//(2*m) for ii in range(m)]
    indices = np.sort(np.hstack([[0, len(x)-1], f(deg-1, len(x))]))
  else:
    assert type(seeds) in [list, np.ndarray], "Expecting a list or numpy array."
    assert len(seeds) == len(np.unique(seeds)), "Expecting `seeds` to have distinct entries."
    assert len(seeds) >= deg+1, "Expecting `seeds` list to have at least {} entries.".format(deg+1)
    indices = seeds
  errors = []
  
  return np.array(indices, dtype='int'), errors


def _greedy(x, y, tol=1e-6, rel=False, deg=5, verbose=False, seeds=None):
  """Greedy algorithm for building a reduced-order spline"""
  if verbose:
    print "\nSize", "\t", "Error"
    print "="*13
  
  if rel:
    ymax = np.max(np.abs(y))
    assert ymax > 0., "All data samples are zero."
  else:
    ymax = 1.
  _tol = tol*ymax
  
  # Seed greedy algorithm
  indices, errors = _seed(x, deg=deg, seeds=seeds)
  args = indices[:]  # Keep track of unsorted indices
  
  # Greedy algorithm
  flag, ctr = 0, len(indices)+1
  while flag == 0 and ctr < len(x):	
    # Spline interpolant on current set of knots
    s = UnivariateSpline(x[indices], y[indices], k=deg, s=0)
    
    # L-infinity errors of current spline interpolant and data
    errs = np.abs( s(x)-y )
    
    # Get the index of the largest interpolation error on y
    imax = np.argmax( errs )
    
    # Update arrays with data point that gives largest interpolation error
    errors.append( errs[imax] )
    args = np.hstack([args, imax])
    indices = np.sort(np.hstack([indices, imax]))  # Indices sorted for spline interpolation
    
    # Print to screen, if requested
    if verbose:
      print ctr, "\t", errors[-1]
    
    # Check if greedy error is below tolerance and exit if so
    if errors[-1] < _tol:
      flag = 1
      args = args[:-1]
      indices = np.sort(args)
      errors = np.array(errors[:-1])
    
    ctr += 1
  
  # Construct the spline from the reduced data
  spline = UnivariateSpline(x[indices], y[indices], k=deg, s=0)
  
  return spline, indices, args, errors, _tol

