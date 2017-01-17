import numpy as np


################################
#   Class for generating some  #
#   test data for showing how  #
#    to use the code in the    #
#   Jupyter/IPython notebooks  #
################################

class TestData(object):
  """Generate the test data used as in example IPython notebooks 
    for demonstrating the construction, properties, and errors 
    of a reduced-order spline interpolant.
  """
  
  def __init__(self, num=4001, noise=0., uv=0.):
    """Create a TestData object.
    
    Input
    -----
      num   -- number of samples to evaluate the function
               in domain [-1,1]
      noise -- amplitude of stochastic fluctuations added
               to smooth function values
               (default is 0.)
      uv    -- amplitude of high-frequency (i.e., ultra-violet)
               features added to smooth function values
               (default is 0.)
    
    Attributes
    ----------
      x -- samples
      y -- values of sampled function
    
    """
    
    # Generate test data
    self.x = np.linspace(-1, 1, num)
    self.y = self.f(self.x, noise=noise, uv=uv)
  
  
  def f(self, x, noise=0., uv=0.):
    """Function to sample for reduced-order spline examples
    
    Inputs
    ------
      x     -- values to sample the (smooth) function
      noise -- amplitude of stochastic fluctuations added
               to smooth function values
               (default is 0.)
      uv    -- amplitude of high-frequency (i.e., ultra-violet)
               features added to smooth function values
               (default is 0.)
    
    Output
    ------
      sampled function values
    
    Comments
    --------
    The function being evaluated is
    
      f(x) = 100.*( (1.+x) * sin(5.*(x-0.2)**2) 
                    + exp(-(x-0.5)**2/2./0.01) * sin(100*x) 
                  )
    """
    
    # Validate inputs
    x = np.asarray(x)
    
    # Return smooth function values
    ans = 100.*( (x+1.)*np.sin(5.*(x-0.2)**2) + np.exp(-(x-0.5)**2/2./0.01)*np.sin(100*x) )
    
    # Return smooth function values with high-frequency (UV) features
    if uv != 0.:
      assert type(uv) in [float, int], "Expecting integer or float type."
      ans += float(uv)*self.uv(x)
    
    # Return smooth function values with stochastic noise
    if noise != 0.:
      assert type(noise) in [float, int], "Expecting integer or float type."
      ans += float(noise)*np.random.randn(len(x))
    
    return ans
  
  
  def dfdx(self, x):
    """Analytic derivative of f
    
    Inputs
    ------
      x -- values to sample the derivative of 
           the function f(x) (see self.f method)
    
    Outputs
    -------
      ans -- values of analytically calculated 
             derivative of the function
    
    """
    x = np.asarray(x)
    
    a = 10.*(-0.2+x)*(1.+x)*np.cos(5.*(-0.2 + x)**2)
    b = 100.*np.exp(-50.*(-0.5+x)**2)*np.cos(100.*x)
    c = np.sin(5.*(-0.2+x)**2)
    d = -100.*np.exp(-50.*(-0.5+x)**2)*(-0.5+x)*np.sin(100.*x)
    
    ans = 100.*(a+b+c+d)
    
    return ans
  
  
  def uv(self, x, width=20):
    """Generate high-frequency oscillations
    
    Inputs
    ------
      x     -- values to sample the high-frequency
               oscillations
      width -- number of samples corresponding to
               the period of the high-frequency
               oscillations
               (default is 20)
    
    Outputs
    -------
      array of high-frequency oscillating values
    
    """
    X = x[width] - x[0]
    return np.sin(len(x)/X * x)


