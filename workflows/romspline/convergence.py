from __init__ import state

if state._MATPLOTLIB:
  import matplotlib.pyplot as plt

import numpy as np
import greedy
from helpers import Linfty



##############################
# Class to study convergence #
#    of greedy algorithm     #
##############################

class Convergence(object):
  """A class for determining if the reduced-order spline greedy algorithm is
  convergent. 
  
  The original data set is decimated and reduced-order splines
  are built for each decimation. Convergence is met if the size of the
  reduced dataset is essentially unchanged for consecutive decimations.
  """
  
  def __init__(self, spline=None, tol=1e-6, rel=False, deg=5, seeds=None):
    """Create a Convergence object.
    
    Input
    -----
      spline -- a greedy.ReducedOrderSpline class object
                (default is None)
      tol    -- L-infinity error tolerance for reduced-order spline 
                (default 1e-6)
      rel    -- L-infinity error tolerance is relative to max abs of data?
                (default False)
      deg    -- degree of interpolating polynomials 
                (default 5)
    """
    
    # Initialize some variables
    self._tol = tol
    self._rel = rel
    self._deg = deg
    self._seeds = seeds
    self.splines = dict()
    self.errors = dict()
    self.compressions = dict()
    
    if spline is None:
      self._spline = False
    else:
      self._spline = spline
      # Add self._made = True?
    self._made = False
  
  
  def __call__(self, x, y, levs=None):
    """Decimate the input data at the required levs and compute the 
    resulting reduced-order spline interpolants.
    
    Input
    -----
      x    -- samples
      y    -- data to be downsampled
      levs -- decimation levels (levs) such that if levs=4 then 
              every fourth entry is skipped for building the spline
              (default None)
    
    Attributes
    ----------
      splines      -- reduced-order spline objects for each lev
      errors       -- L-infinity spline errors for each lev as
                      compared to the full input data
      compressions -- compression factors of the reduced data
                      for each lev
    """
    return self.make(x, y, levs=levs)
  
  
  def make(self, x, y, levs=None):
    """Decimate the input data at the required levs and compute the 
    resulting reduced-order spline interpolants.
    
    Input
    -----
      x    -- samples
      y    -- data to be downsampled
      levs -- decimation levels (levs) such that if levs=4 then 
              every fourth entry is skipped for building the spline
              (default None)
    
    Attributes
    ----------
      splines      -- reduced-order spline objects for each lev
      errors       -- L-infinity spline errors for each lev as
                      compared to the full input data
      compressions -- compression factors of the reduced data
                      for each lev
    """
    if levs is None:
      self.levs = np.array([8, 4, 2])
    else:
      self.levs = np.array(levs)
    if 1 not in self.levs:
      self.levs = np.hstack([self.levs, 1])
    
    # Build reduced-order spline on full data (e.g., lev=1)
    if self._spline:
      self.splines[1] = self._spline
    else:
      self.splines[1] = greedy.ReducedOrderSpline(x, y, deg=self._deg, tol=self._tol, rel=self._rel, seeds=self._seeds)
    
    # Build reduced-order splines for each requested decimation level
    for ll in self.levs:
      
      # Build a reduced_order spline
      if ll != 1:
        x_lev = x[::ll]
        y_lev = y[::ll]
        if x[-1] not in x_lev:
          x_lev = np.hstack([x_lev, x[-1]])
          y_lev = np.hstack([y_lev, y[-1]])
        
        # Adjust the indices of the seeds for the different lev
        if self._seeds is None:
          seeds = None
        else:
          seeds = [int(float(ss)/ll) for ss in self._seeds]
        
        self.splines[ll] = greedy.ReducedOrderSpline(x_lev, y_lev, deg=self._deg, tol=self._tol, rel=self._rel, seeds=seeds)
      
      # Compute the compression factor and L-infinity absolute error for this lev
      self.errors[ll] = Linfty(self.splines[ll].eval(x), y)
      self.compressions[ll] = self.splines[ll].compression
    
    self._made = True
  
  
  def _plot_greedy_errors(self, ax, axes=None):
    """Simple function that selects the plotting axes type
    
    Input
    -----
      ax   -- matplotlib plot/axis object
      axes -- type of axes to plot
              (default None)
    
    Output
    ------
      matplotlib plot/axis object
    """
    
    if axes is None:
      axes = 'semilogy'
    if axes == 'semilogy':
      [ax.semilogy(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'semilogx':
      [ax.semilogx(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'loglog':
      [ax.loglog(self.splines[ll].errors) for ll in self.levs]
    elif axes == 'plot':
      [ax.plot(self.splines[ll].errors) for ll in self.levs]
    
    return ax
  
  
  def plot_greedy_errors(self, ax=None, show=True, axes='semilogy', xlabel='Size of reduced data', ylabel='Greedy errors'):
    """Plot the greedy errors versus size of the reduced data.
    
    Input
    -----
      ax     -- matplotlib plot/axis object
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
    
    if self._made:  # Check if spline data made
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax = self._plot_greedy_errors(ax, axes=axes)
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
      print "No data to plot. Run `make` method."
  
  
  def _plot_Linfty_errors(self, ax):
    
    [ax.loglog(ll, self.errors[ll], 'o') for ll in self.levs]
    
    return ax
  
  
  def plot_Linfty_errors(self, ax=None, show=True, xlabel='Decimation factor', ylabel='Max spline errors'):
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      #[ax.loglog(ll, self.errors[ll], 'o') for ll in self.levs]
      ax = self._plot_Linfty_errors(ax)
      ax.set_xlim(self.levs.min(), 1.1*self.levs.max())
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
      print "No data to plot. Run `make` method."
  
  
  def _plot_sizes(self, ax, axes=None):
    """Simple function that selects the plotting axes for 
    displaying data from the plot_sizes method.
    
    Input
    -----
      ax   -- matplotlib plot/axis object
      axes -- type of axes to plot
              (default None)
    
    Output
    ------
      matplotlib plot/axis object
    """
    
    if axes is None:
      axes = 'semilogx'
    if axes == 'semilogy':
      [ax.semilogy(ll, self.splines[ll].size, 'o') for ll in self.levs]
    elif axes == 'semilogx':
      [ax.semilogx(ll, self.splines[ll].size, 'o') for ll in self.levs]
    elif axes == 'loglog':
      [ax.loglog(ll, self.splines[ll].size, 'o') for ll in self.levs]
    elif axes == 'plot':
      [ax.plot(ll, self.splines[ll].size, 'o') for ll in self.levs]
    
    return ax
  
  
  def plot_sizes(self, ax=None, show=True, axes='semilogx', xlabel='Decimation factor', ylabel='Reduced data sizes'):
    """Plot the size of the reduced-order splines built 
    for each requested decimation of the original dataset
    
    Input
    -----
      ax     -- matplotlib plot/axis object
                (default None)
      show   -- display the plot?
                (default True)
      axes   -- axis scales for plotting
                (default 'semilogx')
      xlabel -- label of x-axis
                (default 'Decimation factor')
      ylabel -- label of y-axis
                (default 'Reduced data sizes')
    
    Output
    ------
      If show=True then the plot is displayed.
      Otherwise, the matplotlib plot/axis object is output.
    """
    
    if self._made:
      
      if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
      
      ax = self._plot_sizes(ax, axes=axes)
      ax.set_xlim(self.levs.min(), 1.1*self.levs.max())
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
      print "No data to plot. Run `make` method."


