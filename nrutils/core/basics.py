#
# from __future__ import print_function
from positive import *
from positive.physics import *

#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
''' Methods/Class for modeled PSDs '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#

# Einstein Telescope
def etb_psd(freq):
    # Analytic formula from arxiv:1005.0304, eq. 2.2 and 2.3

    #
    from numpy import inf,ndarray

    # (eq. 2.3) Fitting Constants
    a1 = 2.39*(10**(-27));   b1 = -15.64
    a2 = 0.349;             b2 = -2.145
    a3 = 1.76;              b3 = -0.12
    a4 = 0.409;             b4 = 1.10
    # -------------------------------- #
    f0 = 100          # Hz
    S0 = 10.0**(-50)  # Hz**-1
    x = freq / f0     # unitless

    # (eq. 2.2) The Analytic Fit
    Sh_f = S0 * \
                ( a1*pow(x,b1) + \
                  a2*pow(x,b2) + \
                  a3*pow(x,b3) + \
                  a4*pow(x,b4) )**2

    # Impose Low Frequency Cut-Off of 1 Hz %
    if isinstance(freq,ndarray):
        mask = freq <= 1
        Sh_f[mask] = inf

    #
    ans = Sh_f
    return ans

# Initial LIGO
def iligo_psd(freq,version=1):
    '''
    Modeled iLIGO noise curves from arxiv:0901.4936 (version=2) and arxiv:0901.1628 (version=1)
    '''
    #
    f0 = 150
    xx = freq/f0
    #
    if version in (2,'0901.4936'):
        # analytic formula from Ajith and Bose: arxiv: 0901.4936 eq 3.1 strain^2 / Hz
        Sn = 9e-46*( (4.49*xx)**(-56) + 0.16*xx**(-4.52) + 0.52 + 0.32*xx**2 )
    else:
        # This is Eq. 9 of https://arxiv.org/pdf/0901.1628.pdf
        Sn = 3.136e-46 * ( (4.449*xx)**-56 + \
                          0.16*xx**-4.52 + \
                          xx*xx + 0.52 )
    # Return answer
    ans = Sn
    return ans

#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
''' Methods for low-level waveform manipulation '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#

# Function that returns true if for string contains l assignment that is less than l_max
def l_test(string,l_max):
    '''
    Function that returns true if for string contains l assignment that is <= l_max:
    score = ltest('Ylm_l3_m4_stuff.asc',3)
          = True
    score = ltest('Ylm_l3_m4_stuff.asc',5)
          = True
    score = ltest('Ylm_l6_m4_stuff.asc',2)
          = False
    '''
    # break string into bits by l
    score = False
    for bit in string.split('l'):
        if bit[0].isdigit():
            score = score or int( bit[0] )<= l_max

    # return output
    return score


# Interpolate waveform array to a given spacing in its first column
def intrp_wfarr(wfarr,delta=None,domain=None,verbose = False):

    #
    from numpy import linspace,array,diff,zeros,arange
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Validate inputs
    if (delta is None) and (domain is None):
        msg = red('First "delta" or "domain" must be given. See traceback above.')
        error(msg,'intrp_wfarr')
    if (delta is not None) and (domain is not None):
        msg = red('Either "delta" or "domain" must be given, not both. See traceback above.')
        error(msg,'intrp_wfarr')

    # Only interpolate if current delta is not input delta
    proceed = True
    if delta is not None:
        d = wfarr[1,0]-wfarr[0,0]
        if verbose: alert('The original dt is %f and the requested on is %f.'%(d,delta))
        if abs(delta-d)/(delta+d) < 1e-6:
            proceed = False
            # warning('The waveform already has the desired time step, and so will not be interpolated.')

    # If there is need to interpolate, then interpolate.
    if proceed:

        #
        if verbose: alert('Proceeding to interpolate to dt = %f.'%delta)

        # Encapsulate the input domain for ease of reference
        input_domain = wfarr[:,0]

        # Generate or parse the new domain
        if domain is None:
            N = diff(lim(input_domain))[0] / delta
            intrp_domain = delta * arange( 0, N  ) + wfarr[0,0]
        else:
            intrp_domain = domain

        # Pre-allocate the new wfarr
        _wfarr = zeros( (len(intrp_domain),wfarr.shape[1]) )

        # Store the new domain
        _wfarr[:,0] = intrp_domain
        if verbose: alert('The new dt is %f'%diff(intrp_domain)[0])

        # Interpolate the remaining columns
        for k in range(1,wfarr.shape[1]):
            _wfarr[:,k] = spline( input_domain, wfarr[:,k] )( intrp_domain )

    else:

        alert('The waveform array will %s be interpolated.'%(bold(red('NOT'))))

        # Otherwise, return the input array
        _wfarr = wfarr

    #
    return _wfarr


# Fucntion to pad wfarr with zeros. NOTE that this should only be applied to a time domain waveform that already begins and ends with zeros.
def pad_wfarr(wfarr,new_length,where=None,verbose=None,extend=True,__nowarn__=False):

    #
    from numpy import hstack,zeros,arange,pad,unwrap,angle,cos,sin


    # NOTE that the waveform array must be uniformly space at this point. This will be handled by straighten_wfarr(()
    wfarr = straighten_wfarr( wfarr, verbose )

    # Only pad if size of the array is to increase
    length = len(wfarr[:,0])
    # Look for option to interpret input as "length to pad" rather than "total new length"
    if extend: new_length+= length
    proceed = length < new_length

    #
    if isinstance(where,str):
        where = where.lower()

    # Warn the user if extend is false
    if not extend:
        msg = 'You have disabled the extend option. As a result the input padding length will be interpreted as the desired total length of the new waveform array. This course is discouraged in favor of e.g. using the fftlength option when taking fouorier transforms, OR simply inputting the desired pad amount.'
        warning(msg, verbose=(not __nowarn__) and verbose)

    #
    if where is None:
        where = 'sides'
    elif not isinstance(where,str):
        error('where must be string: left,right,sides','pad_wfarr')
    elif where not in ['left','right','sides']:
        error('where must be in {left,right,sides}','pad_wfarr')


    # Enforce integer new length
    if new_length != int(new_length):
        msg = 'Input pad length is not integer; I will apply int() before proceeding.'
        alert(msg,'pad_wfarr')
        new_length = int( new_length )

    #
    if proceed:


        # Pre-allocate the new array
        _wfarr = zeros(( new_length, wfarr.shape[1] ))

        # Create the new time series
        dt = wfarr[1,0] - wfarr[0,0]
        _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0]

        if where == 'sides':

            # Create the pads for the other columns
            left_length = int(new_length-length)/2
            right_length = new_length-length-left_length
            left_pad = zeros( left_length )
            right_pad = zeros( right_length )

            _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0] - dt*(left_length-1)

            # # Pad the remaining columns
            # for k in arange(1,wfarr.shape[1]):
            #     _wfarr[:,k] = hstack( [left_pad,wfarr[:,k],right_pad] )
            #     # _wfarr[:,k] = pad( wfarr[:,k], (left_length,right_length), 'linear_ramp'  )

            # Pad amplitude and phase, then restructure
            y = wfarr[:,1] + 1j*wfarr[:,2]
            amp = abs( y )
            pha = unwrap( angle(y) )
            amp_ = pad( amp, (left_length,right_length), 'constant' )
            pha_ = pad( pha, (left_length,right_length), 'edge' )
            _wfarr[:,1] = amp_ * cos(pha_)
            _wfarr[:,2] = amp_ * sin(pha_)

        elif where == 'right':

            _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0]

            # Create the pads for the other columns
            right_pad = zeros( new_length-length )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [wfarr[:,k],right_pad] )

        elif where == 'left':

            _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0] - dt*int(new_length-length-1)

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k]] )

    else:

        # Otherwise, do nothing.
        _wfarr = wfarr

        # Warn the user that nothing has happened.
        if extend and (new_length!=length):
            msg = 'The desired new length is <= the current array length (i.e. number of time domain points). Nothing will be padded. Perhaps you want to set extend=True, and then input the amount which you wish to pad? <3 '
            error( msg,fname='pad_wfarr'+cyan('@%i'%linenum()) )

    #
    if _wfarr.shape[0] != new_length:
        error('The current length (%i) is not the desired new length(%i). This function has a bug.'%(_wfarr.shape[0],new_length))

    #
    if verbose:
        alert('The shape was %s. Now the shape is %s.'%(wfarr.shape,_wfarr.shape) )

    # Return padded array
    # print _wfarr.shape[0],new_length
    # print wfarr.shape, _wfarr.shape
    return _wfarr


#
def plot_wfarr(wfarr,domain=None,show=False,labels=None):

    #
    from matplotlib.pyplot import figure,plot,show,xlabel,ylabel,title

    #
    warning('Method under development.')

    # Plot time domain
    # figure()
    plot( wfarr[:,0], wfarr[:,1] )
    plot( wfarr[:,0], wfarr[:,2] )
    # show()

    # Plot frequency domain

# Shift a waveform arra by some "shift" amount in time
def tshift_wfarr( _wfarr, shift, method=None, verbose=False ):
    '''Shift a waveform array by some "shift" amount in time'''
    # Import useful things
    from numpy import array
    # Unpack waveform array
    t,p,c = _wfarr[:,0],_wfarr[:,1],_wfarr[:,2]
    _y = p + 1j*c
    # Shift the waveform array data using tshift
    y = tshift( t,_y,shift,method=method,verbose=verbose )
    # Repack the input array
    wfarr = array(_wfarr)
    wfarr[:,0] = t
    wfarr[:,1] = y.real
    wfarr[:,2] = y.imag
    # Return answer
    ans = wfarr
    return ans


# Shift phase of waveform array
def shift_wfarr_phase(wfarr,dphi):

    #
    from numpy import array,ndarray,sin,cos

    #
    if not isinstance(wfarr,ndarray):
        error( 'input must be numpy array type' )

    #
    t,r,c = wfarr[:,0],wfarr[:,1],wfarr[:,2]

    #
    r_ = r*cos(dphi) - c*sin(dphi)
    c_ = r*sin(dphi) + c*cos(dphi)

    #
    wfarr[:,0],wfarr[:,1],wfarr[:,2] = t , r_, c_

    #
    return wfarr


# Find the average phase difference and align two wfarr's
def align_wfarr_average_phase(this,that,mask=None,verbose=False):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    #
    from numpy import angle,unwrap,mean,pi,mod

    #
    if mask is None:
        u = this[:,1]+1j*this[:,2]
        v = that[:,1]+1j*that[:,2]
    else:
        u = this[mask,1]+1j*this[mask,2]
        v = that[mask,1]+1j*that[mask,2]

    #
    _a = unwrap( angle(u) )
    _b = unwrap( angle(v) )


    #
    a,b = mean( _a ), mean( _b )
    dphi = -a + b

    #
    if verbose:
        alert('The phase shift applied is %s radians.'%magenta('%1.4e'%(dphi)))

    #
    dphi = mod(dphi,2*pi)
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


# Given a dictionary of multipoles and wafarrs, recompose at a desired theta and phi
def recompose_wfarrs( wfarr_dict, theta, phi ):
    '''
    Given a dictionary of spin -2 spherical harmonic multipoles, recompose at a desired theta and phi:

        recomposed_wfarr = recompose_wfarr( wfarr_dict, theta, phi )

    ---

    Inputs:

     * wfarr_dict: dictionary with keys being (l,m), and values being the related wfarrs (time or frequency domain)
     * theta,phi: the polar and azximuthal angles desired for recomposition

    '''

    # Import useful things
    from numpy import ndarray,zeros,dot,array,zeros_like

    #-%-%-%-%-%-%-%-%-%-%-%-#
    # Validate wfarr_dict   #
    #-%-%-%-%-%-%-%-%-%-%-%-#
    for k in wfarr_dict:
        # keys must be length 2
        if len( k ) != 2:
            error( 'keys must be length 2, and compised of spherical harmonic l and m (e.g. (2,1) )' )
        # elements within key must be integers
        for v in k:
            if not isinstance(v,int):
                error( 'invalid multipole eigenvalue found: %s'%[v] )
        # key values must be ndarray
        if not isinstance(wfarr_dict[k],ndarray):
            alert(wfarr_dict[k].__class__)
            error('key values must be ndarray')

    # Number of samples
    n_samples = wfarr_dict[k].shape[0]
    # Number of multipoles given
    n_multipoles = len( wfarr_dict )

    #
    def __recomp__( column_index ):
        # Create matrices to hold spherical harmonic and waveform array data
        M = zeros( [ n_samples, n_multipoles ], dtype=complex )
        Y = zeros( [ n_multipoles, 1 ], dtype=complex )
        # Seed the matrix as well as the vector of spherical harmonic values
        for k,(l,m) in enumerate(wfarr_dict.keys()):
            wfarr = wfarr_dict[l,m]
            M[:,k] = wfarr[:,column_index]
            Y[k] = sYlm(-2,l,m,theta,phi)
        # Perform the matrix multiplication and create the output gwf object
        Z = dot( M,Y )[:,0] # NOTE that the "[:,0]" is to enforce a shape of (N,1) rather than (N,)
        #
        ans = Z
        return Z

    # Extract time/frequency domain for output
    domain = wfarr_dict[ wfarr_dict.keys()[0] ][:,0]
    # Recompose plus and cross columns separately
    recomposed_plus = __recomp__(1)
    recomposed_cross = __recomp__(2)

    # Construct recomposed wfarr
    recomposed_wfarr = array( [ domain, recomposed_plus, recomposed_cross ] ).T

    # Output answer
    ans = recomposed_wfarr
    return ans


#
def recompose_complex_waveforms( y_dict, theta, phi ):

    # Import useful things
    from numpy import ndarray,zeros,dot,array

    # Number of samples
    n_samples = y_dict[y_dict.keys()[0]].shape[0]
    # Number of multipoles given
    n_multipoles = len( y_dict )

    # Create matrices to hold spherical harmonic and waveform array data
    M = zeros( [ n_samples, n_multipoles ], dtype=complex )
    Y = zeros( [ n_multipoles, 1 ], dtype=complex )
    # Seed the matrix as well as the vector of spherical harmonic values
    for k,(l,m) in enumerate(y_dict.keys()):
        M[:,k] = y_dict[l,m]
        Y[k] = sYlm(-2,l,m,theta,phi)
    # Perform the matrix multiplication and create the output gwf object
    Z = dot( M,Y )[:,0] # NOTE that the "[:,0]" is to enforce a shape of (N,1) rather than (N,)

    #
    ans = Z
    return Z

#
def get_wfarr_relative_phase(this,that, mask=None):

    #
    from numpy import angle,unwrap,mean,ones_like,pi

    #
    if mask is None: mask = ones_like(this[:,0],dtype=bool)

    #
    u = this[mask,1]+1j*this[mask,2]
    v = that[mask,1]+1j*that[mask,2]

    #
    _a = unwrap( angle(u) )[0]
    _b = unwrap( angle(v) )[0]

    #
    dphi = -_a + _b

    #
    return dphi


# Find the average phase difference and align two wfarr's
def align_wfarr_initial_phase(this,that, mask=None ):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    #
    from numpy import pi,mod

    dphi = get_wfarr_relative_phase(this,that,mask=mask)

    #
    dphi = mod(dphi,2*pi)
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


# Fix nans, nonmonotinicities and jumps in time series waveform array
def straighten_wfarr( wfarr, verbose=False ):
    '''
    Some waveform arrays (e.g. from the BAM code) may have non-monotonic time series
    (gaps, duplicates, and crazy backwards referencing). This method seeks to identify
    these instances and reformat the related array. Non finite values will also be
    removed.
    '''

    # Import useful things
    from numpy import arange,sum,array,diff,isfinite,hstack,allclose,median
    thisfun = 'straighten_wfarr'
    
    #
    rw,cl = wfarr.shape 
    if rw == 0:
        error('input waveform array has zero rows')
    if cl == 0:
        error('input waveform array has zero columns')

    # check whether t is monotonically increasing
    isincreasing = allclose( wfarr[:,0], sorted(wfarr[:,0]), 1e-6 )
    if not isincreasing:
        # Let the people know
        msg = red('The time series has been found to be non-monotonic. We will sort the data to enforce monotinicity.')
        if verbose: warning(msg)
        # In this case, we must sort the data and time array
        map_ = arange( len(wfarr[:,0]) )
        map_ = sorted( map_, key = lambda x: wfarr[x,0] )
        wfarr = wfarr[ map_, : ]
        # if allclose( wfarr[:,0], sorted(wfarr[:,0]), 1e-6 ) and verbose: warning(red('The waveform time series is now monotonic.'))

    # Remove rows that contain non-finite data
    finite_mask = isfinite( sum( wfarr, 1 ) )
    # if sum(finite_mask)!=len(finite_mask):
    #     if verbose: warning('Non-finite values found in waveform array. Corresponding rows will be removed.',thisfun)
    wfarr = wfarr[ finite_mask, : ]

    # Sort rows by the time series' values
    time = array( wfarr[:,0] )
    space = arange( wfarr.shape[0] )
    chart = sorted( space, key = lambda k: time[k] )
    # if (space != chart).all():
    #     if verbose: warning('The waveform array was found to have nonmonotinicities in its time series. The array will now be straightened.',thisfun)
    wfarr = wfarr[ chart, : ]

    # Remove rows with duplicate time series values
    time = array( wfarr[:,0] )
    dt = median( diff(time) )
    diff_mask = hstack( [ True, diff(time)/dt>1e-6 ] )
    # if sum(diff_mask)!=len(diff_mask):
    #     if verbose: warning('Repeated time values were found in the array. Offending rows will be removed.',thisfun)
    wfarr = wfarr[ diff_mask, : ]

    # The wfarr should now be straight
    # NOTE that the return here is optional as all operations act on the original input
    return wfarr


#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
# Find the polarization and orbital phase shifts that maximize the real part
# of  gwylm object's (2,2) and (2,1) multipoles at merger (i.e. the sum)
''' See gwylm.selfalign for higher level Implementation '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#

def vectorize( _gwylmo, dphi, dpsi, k_ref=0 ):
    from numpy import array
    vec = []
    select_modes = [ (2,2), (2,1) ]
    valid_count = 0
    gwylmo = _gwylmo.rotate( dphi=dphi, dpsi=dpsi, apply=False, verbose=False, fast=True )
    for y in gwylmo.ylm:
        l,m = y.l,y.m
        if (l,m) in select_modes:
            vec.append( y.plus[ k_ref ] )
            valid_count += 1
    if valid_count != 2:
        error('input gwylm object must have both the l=m=2 and (l,m)=(2,1) multipoles; only %i of these was found'%valid_count)
    return array(vec)

def alphamax(_gwylmo,dphi,plt=False,verbose=False,n=13):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,sum,argmax,array
    action = lambda x: sum( vectorize( _gwylmo, x[0], x[1] ) )
    dpsi_range = linspace(-1,1,n)*pi
    dpsis = linspace(-1,1,1e2)*pi
    a = array( [ action([dphi,dpsi]) for dpsi in dpsi_range ] )
    aspl = spline( dpsi_range, a, kind='cubic' )(dpsis)
    dpsi_opt_guess = dpsis[argmax(aspl)]
    K = minimize( lambda PSI: -action([dphi,PSI]), dpsi_opt_guess )
    dpsi_opt = K.x[-1]
    if plt:
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot, xlabel
        plot( dpsi_range, a, linewidth=4, color='k', alpha=0.1 )
        plot( dpsis, aspl, label=dpsi )
        plot( dpsis[argmax(aspl)], aspl[argmax(aspl)], 'or', mfc='none' )
        xlabel(r'$\psi$')
    if verbose: alert(dpsi_opt,action([dphi,dpsi_opt]))
    return [ dpsi_opt, action([dphi,dpsi_opt])    ]

def betamax(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array
    dphi_list = pi*linspace(-1,1,n)
    dpsi,val = [],[]
    for dphi in dphi_list:
        [dpsi_,val_] = alphamax(_gwylmo,dphi,plt=False,n=n)
        dpsi.append( dpsi_ )
        val.append( val_ )

    dphis = linspace(min(dphi_list),max(dphi_list),1e3)
    vals = spline( dphi_list, val, kind='cubic' )( dphis )
    dpsi_s = spline( dphi_list, dpsi, kind='cubic' )( dphis )

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )
    dphi_opt_guess = dphis[argmax(vals)]
    dpsi_opt_guess = dpsi_s[argmax(vals)]
    if opt:
        K = minimize( action, [dphi_opt_guess,dpsi_opt_guess] )
        # print K
        dphi_opt,dpsi_opt = K.x
        val_max = -K.fun
    else:
        dphi_opt = dphi_opt_guess
        dpsi_opt = dpsi_opt_guess
        val_max = vals.max()

    if plt:
        # Setup plotting backend
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot,xlabel,title
        plot( dphi_list, val, linewidth=4, alpha=0.1, color='k' )
        plot( dphi_opt, val_max, 'or', alpha=0.5 )
        plot( dphis, vals )
        xlabel(r'$\phi$')
        title(val_max)

    # if verbose:
    #     print 'dphi_opt = ' + str(dphi_opt)
    #     print 'dpsi_opt = ' + str(dpsi_opt)
    #     print 'val_max = ' + str(val_max)

    return dphi_opt,dpsi_opt

def betamax2(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )

    dphi,dpsi,done,k = pi,pi/2,False,0
    while not done:
        dpsi_action = lambda _dpsi: action( [dphi,_dpsi] )
        dpsi = minimize( dpsi_action, dpsi, bounds=[(0,2*pi)] ).x[0]
        dphi_action = lambda _dphi: action( [_dphi,dpsi] )
        dphi = minimize( dphi_action, dphi, bounds=[(0,2*pi)] ).x[0]
        done = k>n
        alert(dphi,dpsi,action([dphi,dpsi]))
        k+=1

    return dphi,dpsi


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Define wrapper for LAL version of PhneomHM/D -- PHYSICAL UNITS
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
def lalphenom( eta,                     # symmetric mass ratio
               M,                       # Total mass in M_Solar
               x1,                      # dimensioless spin1z
               x2,                      # dimensionles spin2z
               theta,                   # source inclination
               phi,                     # source orbital phase
               D,                       # source distance Mpc
               df_phys,                 # frequency spacing Hz
               fmin,                    # min freq Hz
               fmax,                    # max freq Hz
               approx=None,             # Approximant name, default is IMRPhenomHM
               interface_version=None,
               verbose = False):

    #
    import lal
    from numpy import arange,hstack,array,vstack
    import lalsimulation as lalsim
    from lalsimulation import SimInspiralFD, SimInspiralGetApproximantFromString, SimInspiralChooseFDWaveform
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from nrutils import eta2q

    #
    chi1 = [0,0,float(x1)]
    chi2 = [0,0,float(x2)]

    #
    apx ='IMRPhenomHM' if approx is None else approx
    # print apx, lalsim.__dict__[apx]
    # Standardize input mass ratio and convert to component masses
    M_phys = M; q = eta2q(float(eta)); q = max( [q,1.0/q] )
    # NOTE m1>m2 convention
    m2 = M_phys * 1.0 / (1.0+q); m1 = float(q) * m2
    #
    fmin_phys = fmin
    fmax_phys = fmax
    #
    S1 = array(chi1); S2 = array(chi2)
    #
    M_total_phys = (m1+m2) * lal.MSUN_SI
    r = 1e6*D*lal.PC_SI

    #
    FD_arguments = {    'phiRef': phi,
                        'deltaF': df_phys,
                        'f_min': fmin_phys,
                        'f_max': fmax_phys,
                        'm1': m1 * lal.MSUN_SI,
                        'm2' : m2 * lal.MSUN_SI,
                        'S1x' : S1[0],
                        'S1y' : S1[1],
                        'S1z' : S1[2],
                        'S2x' : S2[0],
                        'S2y' : S2[1],
                        'S2z' : S2[2],
                        'f_ref': 0,
                        'r': r,
                        'i': theta,
                        'lambda1': 0,
                        'lambda2': 0,
                        'waveFlags': None,
                        'nonGRparams': None,
                        'amplitudeO': -1,
                        'phaseO': -1,
                        'approximant': lalsim.__dict__[apx] }

    # Use lalsimulation to calculate plus and cross in lslsim dataformat
    hp_lal, hc_lal  = SimInspiralChooseFDWaveform(**FD_arguments) # SimInspiralFD

    hp_ = hp_lal.data.data
    hc_ = hc_lal.data.data
    #
    _hp = array(hp_[::-1]).conj()
    hp = hstack( [ _hp , hp_[1:] ] ) # NOTE: Do not keep duplicate zero frequency point
    #
    _hc = (array(hc_)[::-1]).conj()
    hc = hstack( [ _hc , hc_[1:] ] )
    #
    f_ = arange(hp_lal.data.data.size) * hp_lal.deltaF
    _f = -array(f_[::-1])
    f = hstack( [ _f, f_[1:] ] )
    #
    wfarr = vstack( [ f, hp, hc ] ).T
    # only keep frequencies of interest
    # NOTE that frequencies below fmin ar kept to maintain uniform spacing of the frequency domain
    mask = abs(f) <= fmax
    wfarr = wfarr[mask,:]
    #
    if  abs( hp_lal.deltaF - df_phys ) > 1e-10:
        error('for some reason, df values are not as expected')
    #
    return wfarr

#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
''' Convert dictionary of wavform data into gwylm '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
def dict2gwylm( multipole_dictionary ):
    '''
    This function is to help create gwylm objects without use of the nrutils' simulation catalog.
    The desired input is to be a dictionary of spin -2 weighted spherical multipole moments:

    multipole_dictionary = {  'psi4' :
                                { (2,2):waveform_ndarray, (2,-2):another_waveform_ndarray, ... },
                              'news' :
                                { (2,2):waveform_ndarray, (2,-2):another_waveform_ndarray, ... },
                              'strain' :
                                { (2,2):waveform_ndarray, (2,-2):another_waveform_ndarray, ... },
                            }

    The at least one of the high-level keys (e.g. 'psi4') must exist.

    THIS FUNCTION IS UNDER DEVELOPMENT AND MAY NOT HAVE ALL OF THE FEATURES YOU WANT. :-)
    '''

    #
    error('This function is in development.')

    #
    from nrutils import scentry,gwylm,gwf
    from numpy import inf,zeros

    #
    e = scentry( None, None )
    chi1,chi2 = zeros(3),zeros(3)
    e.S1,e.S2 = zeros(3),zeros(3)

    #
    e.xf,e.mf = 0,0
    e.default_extraction_par = inf
    e.default_level = None
    e.config = None
    e.setname = 'None'
    e.label = 'None'

    # Use shorthand
    md = multipole_dictionary
    # Validate input
    if isinstance(md,dict):
        #
        None
    else:
        #
        error('input must be dicionary')

    #
    if 'strain' in md:

        #
        strain_dict = md['strain']

        #

#Determine if input is a memeber of the gwf class
def isgwf( obj ):
    '''Determine if input is a memeber of the gwf class'''
    if isinstance(obj,object):
        return obj.__class__.__name__=='gwf'
    else:
        return False
#Determine if input is a memeber of the gwylm class
def isgwylm( obj ):
    '''Determine if input is a memeber of the gwylm class'''
    if isinstance(obj,object):
        return obj.__class__.__name__=='gwylm'
    else:
        return False


#
def kind2texlabel(kind):
    '''
    Convert __disc_data_kind__ to latex for internal labeling of plots. See methods in gwylm class
    '''
    if kind=='psi4':
        return '\psi'
    elif kind=='strain':
        return 'h'
    elif kind=='news':
        return 'f'
    else:
        error('unknown __disc_data_kind__ of %s'%red(kind))

#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
''' Low level functions for rotating waveforms '''
# https://arxiv.org/pdf/1304.3176.pdf
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#


# Calculate the emission tensor given a dictionary of multipole data
def calc_Lab_tensor( multipole_dict ):

    '''
    Given a dictionary of multipole moments (single values or time series)
    determine the emission tensor, <L(aLb)>.

    The input must be a dictionary of the format:
    { (2,2):wf_data22, (2,1):wf_data21, ... (l,m):wf_datalm }

    Key referece: https://arxiv.org/pdf/1304.3176.pdf
    Secondary ref: https://arxiv.org/pdf/1205.2287.pdf

    Lionel London 2017
    '''

    # Import usefuls
    from numpy import sqrt,zeros_like,ndarray,zeros,double

    # Rename multipole_dict for short-hand
    y = multipole_dict

    # Allow user to input real and imag parts separately -- this helps with sanity checks
    if isinstance( y[2,2], dict ):
        #
        if not ( ('real' in y[2,2]) and ('imag' in y[2,2]) ):
            error('You\'ve entered a multipole dictionary with separate real and imaginary parts. This must be formatted such that y[2,2]["real"] gives the real part and ...')
        #
        x = {}
        lmlist = y.keys()
        for l,m in lmlist:
            x[l,m]        = y[l,m]['real'] + 1j*y[l,m]['imag']
            x[l,m,'conj'] = x[l,m].conj()
    elif isinstance( y[2,2], (float,int,complex,ndarray) ):
        #
        x = {}
        lmlist = y.keys()
        for l,m in lmlist:
            x[l,m]        = y[l,m]
            x[l,m,'conj'] = y[l,m].conj()
    #
    y = x


    # Check type of dictionary values and pre-allocate output
    if isinstance( y[2,2], (float,int,complex) ):
        L = zeros( (3,3), dtype=complex )
    elif isinstance( y[2,2], ndarray ):
        L = zeros( (3,3,len(y[2,2])), dtype=complex )
    else:
        error('Dictionary values of handled type; must be float or array')

    # define lambda function for useful coeffs
    c = lambda l,m: sqrt( l*(l+1) - m*(m+1) ) if abs(m)<=l else 0

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Compute tensor elements (Eqs. A1-A2 of https://arxiv.org/pdf/1304.3176.pdf)
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # Pre-allocate elements
    I0,I1,I2,Izz = zeros_like(y[2,2]), zeros_like(y[2,2]), zeros_like(y[2,2]), zeros_like(y[2,2])

    # Sum contributions from input multipoles
    for l,m in lmlist:

        # Eq. A2c
        I0 += 0.5 * ( l*(l+1)-m*m ) * y[l,m] * y[l,m,'conj']

        # Eq. A2b
        I1 += c(l,m) * (m+0.5) * ( y[l,m+1,'conj'] if (l,m+1) in y else 0 ) * y[l,m]

        # Eq. A2a
        I2 += 0.5 * c(l,m) * c(l,m+1) * y[l,m] * ( y[l,m+2,'conj'] if (l,m+2) in y else 0 )

        # Eq. A2d
        Izz += m*m * y[l,m] * y[l,m,'conj']

    # Compute the net power (amplitude squared) of the multipoles
    N = sum( [ y[l,m] * y[l,m,'conj'] for l,m in lmlist ] ).real

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Populate the emission tensor ( Eq. A2e )
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # Populate asymmetric elements
    L[0,0] = I0 + I2.real
    L[0,1] = I2.imag
    L[0,2] = I1.real
    L[1,1] = I0 - I2.real
    L[1,2] = I1.imag
    L[2,2] = Izz
    # Populate symmetric elements
    L[1,0] = L[0,1]
    L[2,0] = L[0,2]
    L[2,1] = L[1,2]

    # Normalize
    N[ N==0 ] = min( N[N>0] )
    L = L.real / N

    #
    return L


# Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame
def calc_coprecessing_angles(multipole_dict,       # Dict of multipoles { ... l,m:data_lm ... }
                             domain_vals=None,   # The time or freq series for multipole data
                             ref_orientation=None,  # e.g. initial J; used for breaking degeneracies in calculation
                             return_xyz=False,
                             safe_domain_range=None,
                             transform_domain=None,
                             verbose=None):
    '''
    Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame
    Key referece: https://arxiv.org/pdf/1304.3176.pdf
    Secondary ref: https://arxiv.org/pdf/1205.2287.pdf
    INPUT
    ---
    multipole_dict,       # dict of multipoles { ... l,m:data_lm ... }
    t,                    # The time series corresponding to multipole data; needed
                            only to calculate gamma; Optional
    verbose,              # Toggle for verbosity
    OUTPUT
    ---
    alpha,beta,gamma euler angles as defined in https://arxiv.org/pdf/1205.2287.pdf
    AUTHOR
    ---
    Lionel London (spxll) 2017
    '''

    # Import usefuls
    from scipy.linalg import eig, norm
    from scipy.integrate import cumtrapz
    from numpy import arctan2, sin, arcsin, pi, ones, arccos, double, array
    from numpy import unwrap, argmax, cos, array, sqrt, sign, argmin, round, median

    # Handle optional input
    if ref_orientation is None:
        ref_orientation = ones(3)
        
    #
    # if (transform_domain in None) or (not isinstance(transform_domain,str)):
    #     error('transform_domain keyword input is required and must be in ("td","fd").')
    # if transform_domain.lower() in ('td','time','time_domain'):
    #     IS_FD = False
    # elif transform_domain.lower() in ('fd','freq','frequency_domain'):
    #     IS_FD = True

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Enforce that multipole data is array typed with a well defined length
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    y = multipole_dict
    for l, m in y:
        if isinstance(y[l, m], (float, int)):
            y[l, m] = array([y[l, m], ])
        else:
            if not isinstance(y[l, m], dict):
                # Some input validation
                if domain_vals is None:
                    error(
                        'Since your multipole data is a series, you must also input the related domain_vals (i.e. times or frequencies) array')
                if len(domain_vals) != len(y[l, m]):
                    error('domain_vals array and multipole data not of same length')

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Calculate the emission tensor corresponding to the input data
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    L = calc_Lab_tensor(multipole_dict)

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
    # Compute the eigenvectors and values of this tensor
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

    # NOTE that members of L have the same length as each y[l,m]; the latter has been
    # forced to always have a length above

    # Initialize idents for angles. NOTE that gamma will be handled below
    alpha, beta = [], []
    X, Y, Z = [], [], []

    #
    # reference_z_scale = None
    old_dom_dex = None

    # For all multipole instances
    ref_x, ref_y, ref_z = None, None, None
    flip_z_convention = False
    for k in range(len(L[0, 0, :])):

        # Select the emission matrix for this instance, k
        _L = L[:, :, k]

        # Compute the eigen vals and vecs for this instance
        vals, vec = eig(_L)

        # Find the dominant direction's index
        dominant_dex = argmax(vals)
        if old_dom_dex is None:
            old_dom_dex = dominant_dex
        if old_dom_dex != dominant_dex:
            # print dominant_dex
            old_dom_dex = dominant_dex

        # Select the corresponding vector
        dominant_vec = vec[:, dominant_dex]

        # There is a z axis degeneracy that we will break here
        # by imposing that the z component is always consistent with the initial L

        if not flip_z_convention:
            if sign(dominant_vec[-1]) == -sign(ref_orientation[-1]):
                dominant_vec *= -1
        else:
            if sign(dominant_vec[-1]) == sign(ref_orientation[-1]):
                dominant_vec *= -1

        # dominant_vec *= sign(domain_vals[k])

        # Extract the components of the dominant eigenvector
        _x, _y, _z = dominant_vec

        # Store reference values if they are None
        if ref_x == None:
            ref_x = _x
            ref_y = _y
            ref_z = _z
        else:
            if (ref_x*_x < 0) and (ref_y*_y < 0):
                _x *= -1
                _y *= -1
                _x *= -1

        # Store unit components for reference in the next iternation
        ref_x = _x
        ref_y = _y
        ref_z = _z

        # Look for and handle trivial cases
        if abs(_x)+abs(_y) < 1e-8:
            _x = _y = 0

        #
        X.append(_x)
        Y.append(_y)
        Z.append(_z)

    # Look for point reflection in X
    X = reflect_unwrap(array(X))
    Y = array(Y)
    Z = array(Z)

    # 3-point vector reflect unwrapping
    # print safe_domain_range
    tol = 0.1
    if safe_domain_range is None:
        safe_domain_range = lim(abs(domain_vals))
    safe_domain_range = array(safe_domain_range)
    from numpy import arange, mean
    for k in range(len(X))[1:-1]:
        if k > 0 and k < (len(domain_vals)-1):

            if (abs(domain_vals[k]) > min(abs(safe_domain_range))) and (abs(domain_vals[k]) < max(abs(safe_domain_range))):

                left_x_has_reflected = abs(X[k]+X[k-1]) < tol*abs(X[k-1])
                left_y_has_reflected = abs(Y[k]+Y[k-1]) < tol*abs(X[k-1])

                right_x_has_reflected = abs(X[k]+X[k+1]) < tol*abs(X[k])
                right_y_has_reflected = abs(Y[k]+Y[k+1]) < tol*abs(X[k])

                x_has_reflected = right_x_has_reflected or left_x_has_reflected
                y_has_reflected = left_y_has_reflected or right_y_has_reflected

                if x_has_reflected and y_has_reflected:

                    # print domain_vals[k]

                    if left_x_has_reflected:
                        X[k:] *= -1
                    if right_x_has_reflected:
                        X[k+1:] *= -1

                    if left_y_has_reflected:
                        Y[k:] *= -1
                    if right_y_has_reflected:
                        Y[k+1:] *= -1

                    Z[k:] *= -1

    # Make sure that imag parts are gone
    X = double(X)
    Y = double(Y)
    Z = double(Z)

    #
    IS_FD = (0.5 == round(float(sum(domain_vals > 0))/len(domain_vals), 2))

    #################################################
    # Reflect Y according to nrutils conventions    #
    # NOTE that this step is needed due to the      #
    # m_relative_sign_convention                    #
    Y *= -1                         
    #################################################
    if IS_FD:
        alert('The domain values seem evenly split between positive and negative values. Thus, we will interpret the input as corresponding to ' +
              green('FREQUENCY DOMAIN')+' data.')
    else:
        alert('The domain values seem unevenly split between positive and negative values. Thus, we will interpret the input as corresponding to '+green('TIME DOMAIN')+' data.')

    a = array(ref_orientation)/norm(ref_orientation)
    B = array([X, Y, Z]).T
    b = (B.T/norm(B, axis=1))
    
    # Here we define a test quantity that is always sensitive to each dimension. NOTE that a simple dot product does not have this property if eg a component of the reference orientation is zero. There is likely a better solution here.
    test_quantity = sum( [ a[k]*b[k] if a[k] else b[k] for k in range(3) ] )

    if IS_FD:

        k = domain_vals > 0
        mask = (domain_vals >= min(safe_domain_range)) & (
            domain_vals <= max(safe_domain_range))
        if (test_quantity[mask][0]) < 0:
            X[k] = -X[k]
            Y[k] = -Y[k]
            Z[k] = -Z[k]
        mask = (-domain_vals >= min(safe_domain_range)
                ) & (-domain_vals <= max(safe_domain_range))

        k = domain_vals < 0
        if 1*(test_quantity[mask][0]) < 0:
            X[k] = -X[k]
            Y[k] = -Y[k]
            Z[k] = -Z[k]

        # Finally, flip negative frequency values consistent with real-valued plus and cross waveforms in the co-precessing frame. In some or most cases, this step simply reverses the above.
        from numpy import median
        mask = (abs(domain_vals) >= min(safe_domain_range)) & (
            abs(domain_vals) <= max(safe_domain_range))

        k = domain_vals[mask] < 0
        kp = domain_vals[mask] >= 0

        if sign(median(Z[mask][k])) != -sign(median(Z[mask][kp])):
            k_ = domain_vals < 0
            X[k_] = -X[k_]
            Y[k_] = -Y[k_]
            Z[k_] = -Z[k_]

        if (sign(median(Z[mask][k])) == -sign(ref_orientation[-1])):
            k_ = domain_vals < 0
            X[k_] = -X[k_]
            Y[k_] = -Y[k_]
            Z[k_] = -Z[k_]
        if (sign(median(Z[mask][kp])) == -sign(ref_orientation[-1])):
            kp_ = domain_vals < 0
            X[kp_] = -X[kp_]
            Y[kp_] = -Y[kp_]
            Z[kp_] = -Z[kp_]

        safe_positive_mask = (domain_vals >= min(safe_domain_range)) & (
            domain_vals <= max(safe_domain_range))
        safe_negative_mask = (-domain_vals >= min(safe_domain_range)) & (
            -domain_vals <= max(safe_domain_range))

        Xp = X[safe_positive_mask]
        Yp = Y[safe_positive_mask]
        Zp = Z[safe_positive_mask]

        mask = safe_negative_mask
        Xm = X[mask][::-1]
        Ym = Y[mask][::-1]
        Zm = Z[mask][::-1]

        another_test_quantity = sign(median(Xp*Xm + Yp*Ym + Zp*Zm))
        if another_test_quantity == -1:
            if Zp[0]*ref_orientation[-1] < 0:
                X[domain_vals > 0] *= -1
                Y[domain_vals > 0] *= -1
                Z[domain_vals > 0] *= -1
            if Zm[0]*ref_orientation[-1] < 0:
                X[domain_vals < 0] *= -1
                Y[domain_vals < 0] *= -1
                Z[domain_vals < 0] *= -1

    else:

        mask = (domain_vals >= min(safe_domain_range)) & (
            domain_vals <= max(safe_domain_range))
        if 1*(test_quantity[mask][0]) < 0:
            warning('flipping manually for negative domain')
            X = -X
            Y = -Y
            Z = -Z

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    #                          Calculate Angles                           #
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    alpha = arctan2(Y, X)
    beta = arccos(Z)

    # Make sure that angles are unwrapped
    alpha = unwrap(alpha)
    beta = unwrap(beta)

    # Calculate gamma (Eq. A4 of of arxiv:1304.3176)
    if len(alpha) > 1:
        k = 1
        # NOTE that spline_diff and spline_antidiff live in positive.maths
        gamma = - spline_antidiff(domain_vals, cos(beta)
                                  * spline_diff(domain_vals, alpha, k=k), k=k)
        gamma = unwrap(gamma)
        # Enforce like integration constant for neg and positive frequency gamma; this assumes time series will not have negative values (i.e. the code should work for TD and FD cases)
        neg_mask = domain_vals < 0
        _mask = (-domain_vals) > 0.01
        mask_ = domain_vals > 0.01
        if sum(neg_mask):
            gamma[neg_mask] = gamma[neg_mask] - \
                gamma[_mask][-1] + gamma[mask_][0]
    else:
        # NOTE that this is the same as above, but here we're choosing an integration constant such that the value is zero. Above, no explicit integration constant is chosen.
        gamma = 0

    # Return answer
    if return_xyz == 'all':
        #
        return alpha, beta, gamma, X, Y, Z
    elif return_xyz:
        #
        return X, Y, Z
    else:
        return alpha, beta, gamma



# # Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame
# def calc_coprecessing_angles( multipole_dict,       # Dict of multipoles { ... l,m:data_lm ... }
#                               domain_vals = None,   # The time or freq series for multipole data
#                               ref_orientation = None, # e.g. initial J; used for breaking degeneracies in calculation
#                               return_xyz = False,
#                               safe_domain_range = None,
#                               verbose = None ):

#     '''
#     Given a dictionary of multipole data, calculate the Euler angles corresponding to a co-precessing frame

#     Key referece: https://arxiv.org/pdf/1304.3176.pdf
#     Secondary ref: https://arxiv.org/pdf/1205.2287.pdf

#     INPUT
#     ---
#     multipole_dict,       # dict of multipoles { ... l,m:data_lm ... }
#     t,                    # The time series corresponding to multipole data; needed
#                             only to calculate gamma; Optional
#     verbose,              # Toggle for verbosity

#     OUTPUT
#     ---
#     alpha,beta,gamma euler angles as defined in https://arxiv.org/pdf/1205.2287.pdf

#     AUTHOR
#     ---
#     Lionel London (spxll) 2017
#     '''

#     # Import usefuls
#     from scipy.linalg import eig,norm
#     from scipy.integrate import cumtrapz
#     from numpy import arctan2,sin,arcsin,pi,ones,arccos,double,array
#     from numpy import unwrap,argmax,cos,array,sqrt,sign,argmin,round,median,mean

#     # Handle optional input
#     if ref_orientation is None: ref_orientation = ones(3)

#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
#     # Enforce that multipole data is array typed with a well defined length
#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
#     y = multipole_dict
#     for l,m in y:
#         if isinstance( y[l,m], (float,int) ):
#             y[l,m] = array( [ y[l,m], ] )
#         else:
#             if not isinstance(y[l,m],dict):
#                 # Some input validation
#                 if domain_vals is None: error( 'Since your multipole data is a series, you must also input the related domain_vals (i.e. times or frequencies) array' )
#                 if len(domain_vals) != len(y[l,m]): error('domain_vals array and multipole data not of same length')


#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
#     # Calculate the emission tensor corresponding to the input data
#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
#     L = calc_Lab_tensor( multipole_dict )

#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#
#     # Compute the eigenvectors and values of this tensor
#     #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-#

#     # NOTE that members of L have the same length as each y[l,m]; the latter has been
#     # forced to always have a length above

#     # Initialize idents for angles. NOTE that gamma will be handled below
#     alpha,beta = [],[]
#     X,Y,Z = [],[],[]

#     #
#     # reference_z_scale = None
#     old_dom_dex = None

#     # For all multipole instances
#     ref_x,ref_y,ref_z = None,None,None
#     flip_z_convention = False
#     for k in range( len(L[0,0,:]) ):

#         # Select the emission matrix for this instance, k
#         _L = L[:,:,k]

#         # Compute the eigen vals and vecs for this instance
#         vals,vec = eig( _L )

#         # Find the dominant direction's index
#         dominant_dex = argmax( vals )
#         if old_dom_dex is None: old_dom_dex = dominant_dex
#         if old_dom_dex != dominant_dex:
#             # print dominant_dex
#             old_dom_dex = dominant_dex

#         # Select the corresponding vector
#         dominant_vec = vec[ :, dominant_dex ]

#         # There is a z axis degeneracy that we will break here
#         # by imposing that the z component is always consistent with the initial L

#         if not flip_z_convention:
#             if sign(dominant_vec[-1]) == -sign(ref_orientation[-1]): dominant_vec *= -1
#         else:
#             if sign(dominant_vec[-1]) ==  sign(ref_orientation[-1]): dominant_vec *= -1

#         # dominant_vec *= sign(domain_vals[k])

#         # Extract the components of the dominant eigenvector
#         _x,_y,_z = dominant_vec

#         # Store reference values if they are None
#         if ref_x==None:
#             ref_x = _x
#             ref_y = _y
#             ref_z = _z
#         else:
#             if (ref_x*_x < 0) and (ref_y*_y < 0):
#                 _x *= -1
#                 _y *= -1
#                 _x *= -1

#         # Store unit components for reference in the next iternation
#         ref_x = _x
#         ref_y = _y
#         ref_z = _z

#         # Look for and handle trivial cases
#         if abs(_x)+abs(_y) < 1e-8 :
#             _x = _y = 0

#         #
#         X.append(_x);Y.append(_y);Z.append(_z)

#     # Look for point reflection in X
#     #
#     # c = 2*pi 
#     # X,Y,Z = [ unwrap( c*array(Q) )/c for Q in (X,Y,Z) ]
        
#     from numpy import isnan
#     for Q in [X,Y,Z]:
#         if sum(isnan(Q)):
#             error('them nans')
        
#     X = array(X,dtype=double)
#     Y = array(Y,dtype=double)
#     Z = array(Z,dtype=double)

#     # 3-point vector reflect unwrapping
#     # print safe_domain_range
#     tol = 0.1
#     if safe_domain_range is None: safe_domain_range = lim(abs(domain_vals))
#     safe_domain_range = array( safe_domain_range )
#     from numpy import arange,mean
#     for k in range(len(X))[1:-1]:
#         if k>0 and k<(len(domain_vals)-1):

#             if (abs(domain_vals[k])>min(abs(safe_domain_range))) and (abs(domain_vals[k])<max(abs(safe_domain_range))):

#                 left_x_has_reflected = abs(X[k]+X[k-1])<tol*abs(X[k-1])
#                 left_y_has_reflected = abs(Y[k]+Y[k-1])<tol*abs(X[k-1])

#                 right_x_has_reflected = abs(X[k]+X[k+1])<tol*abs(X[k])
#                 right_y_has_reflected = abs(Y[k]+Y[k+1])<tol*abs(X[k])

#                 x_has_reflected = right_x_has_reflected or left_x_has_reflected
#                 y_has_reflected = left_y_has_reflected or right_y_has_reflected

#                 if x_has_reflected and y_has_reflected:

#                     # print domain_vals[k]

#                     if left_x_has_reflected:
#                         X[k:] *=-1
#                     if right_x_has_reflected:
#                         X[k+1:] *= -1

#                     if left_y_has_reflected:
#                         Y[k:] *=-1
#                     if right_y_has_reflected:
#                         Y[k+1:] *= -1

#                     Z[k:] *= -1

#     #
#     IS_FD = ( 0.5 == round(float(sum(domain_vals>0))/len(domain_vals),2) )
#     if IS_FD:
#         alert('The domain values seem evenly split between positive and negative values. Thus, we will interpret the input as corresponding to '+green('FREQUENCY DOMAIN')+' data.')
#     else:
#         alert('The domain values seem unevenly split between positive and negative values. Thus, we will interpret the input as corresponding to '+green('TIME DOMAIN')+' data.')
    

#     #################################################
#     # Reflect Y according to nrutils conventions    #
#     Y = -Y                                          #
#     #################################################

#     a = array(ref_orientation)/norm(ref_orientation)
#     B = array([X,Y,Z]).T
#     b = (B.T/norm(B,axis=1)).T
#     xb,yb,zb = b.T
#     test_quantity = a[0]*xb+a[1]*yb+a[2]*zb

#     if IS_FD:

#         k = domain_vals>0
#         mask = (domain_vals>=min(safe_domain_range)) & (domain_vals<=max(safe_domain_range))
#         if (test_quantity[mask][0])<0:
#             X[k] = -X[k]
#             Y[k] = -Y[k]
#             Z[k] = -Z[k]
#         mask = (-domain_vals>=min(safe_domain_range)) & (-domain_vals<=max(safe_domain_range))

#         k = domain_vals<0
#         if 1*(test_quantity[mask][0])<0:
#             X[k] = -X[k]
#             Y[k] = -Y[k]
#             Z[k] = -Z[k]

#         # Finally, flip negative frequency values consistent with real-valued plus and cross waveforms in the co-precessing frame. In some or most cases, this step simply reverses the above.
#         from numpy import median
#         mask = (abs(domain_vals)>=min(safe_domain_range)) & (abs(domain_vals)<=max(safe_domain_range))

#         k = domain_vals[mask] < 0
#         kp = domain_vals[mask] >= 0

#         if sign(median(Z[mask][k])) != -sign(median(Z[mask][kp])):
#             k_ = domain_vals < 0
#             X[k_] = -X[k_]
#             Y[k_] = -Y[k_]
#             Z[k_] = -Z[k_]
            
#         if (sign(median(Z[mask][k])) == -sign(ref_orientation[-1])):
#             k_ = domain_vals < 0
#             X[k_] = -X[k_]
#             Y[k_] = -Y[k_]
#             Z[k_] = -Z[k_]
#         if (sign(median(Z[mask][kp])) == -sign(ref_orientation[-1])):
#             kp_ = domain_vals < 0
#             X[kp_] = -X[kp_]
#             Y[kp_] = -Y[kp_]
#             Z[kp_] = -Z[kp_]

#         safe_positive_mask = (domain_vals >= min(safe_domain_range) ) & (domain_vals <= max(safe_domain_range))
#         safe_negative_mask = (-domain_vals >= min(safe_domain_range)) & (
#             -domain_vals <= max(safe_domain_range))
            
#         Zp = Z[ safe_positive_mask ]
#         Zm = Z[ safe_negative_mask ][::-1]
            
#         def calc_another_test_quantity(XX,YY,ZZ):
            
#             safe_positive_mask = (domain_vals >= min(safe_domain_range) ) & (domain_vals <= max(safe_domain_range))
#             safe_negative_mask = (-domain_vals >= min(safe_domain_range)) & (
#             -domain_vals <= max(safe_domain_range))
            
#             Xp = XX[ safe_positive_mask ]
#             Yp = YY[ safe_positive_mask ]
#             Zp = ZZ[ safe_positive_mask ]
            
#             mask = safe_negative_mask
#             Xm = XX[mask][::-1]
#             Ym = YY[mask][::-1]
#             Zm = ZZ[mask][::-1]
            
#             return sign( median( Xp*Xm + Yp*Ym + Zp*Zm ) )
            
#         another_test_quantity = calc_another_test_quantity(X,Y,Z)
#         if another_test_quantity == -1:
#             if Zp[0]*ref_orientation[-1] < 0:
#                 X[domain_vals > 0] *= -1
#                 Y[domain_vals > 0] *= -1
#                 Z[domain_vals > 0] *= -1
#             if Zm[0]*ref_orientation[-1] < 0:
#                 X[domain_vals < 0] *= -1
#                 Y[domain_vals < 0] *= -1
#                 Z[domain_vals < 0] *= -1
#         if calc_another_test_quantity(X, Y, Z) == -1:
#             error('unable to correctly reflect positive and negative frequency ends')

#     else:

#         mask = (domain_vals>=min(safe_domain_range)) & (domain_vals<=max(safe_domain_range))
#         if 1*(test_quantity[mask][0])<0:
#             warning('flipping manually for negative domain')
#             X = -X
#             Y = -Y
#             Z = -Z

#     # One more unwrap step
#     if not IS_FD:
#         c = 2*pi
#         X,Y,Z = [ unwrap(c*Q)/c for Q in (X,Y,Z) ]
#         R = sqrt( X**2 + Y**2 + Z**2 )
#         X,Y,Z = [ Q/R for Q in (X,Y,Z) ]

#     # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
#     #                          Calculate Angles                           #
#     # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

#     alpha = arctan2(Y,X)
#     beta  = arccos(Z)

#     # Make sure that angles are unwrapped
#     alpha = unwrap( alpha )
#     beta  = unwrap( beta  )

#     # Calculate gamma (Eq. A4 of of arxiv:1304.3176)
#     if len(alpha) > 1 :
#         k = 1
#         # NOTE that spline_diff and spline_antidiff live in positive.maths
#         gamma = - spline_antidiff( domain_vals, cos(beta) * spline_diff(domain_vals,alpha,k=k), k=k  )
#         gamma = unwrap( gamma )
#         # Enforce like integration constant for neg and positive frequency gamma
#         if IS_FD:
            
#             safe_positive_mask = (domain_vals >= min(safe_domain_range) ) & (domain_vals <= max(safe_domain_range))
#             safe_negative_mask = (-domain_vals >= min(safe_domain_range)) & (
#             -domain_vals <= max(safe_domain_range))
            
#             gamma[safe_negative_mask] = gamma[safe_negative_mask]-mean(gamma[safe_negative_mask]) + mean(gamma[safe_positive_mask])
            
#             # neg_mask = domain_vals<0
#             # _mask = (-domain_vals)>0.01
#             # mask_ =  domain_vals>0.01
#             # if sum(neg_mask):
#             #     gamma[neg_mask] = gamma[neg_mask] - gamma[_mask][-1] + gamma[mask_][0]
#     else:
#         # NOTE that this is the same as above, but here we're choosing an integration constant such that the value is zero. Above, no explicit integration constant is chosen.
#         gamma = 0

#     # Return answer
#     if return_xyz == 'all':
#         #
#         return alpha,beta,gamma,X,Y,Z
#     elif return_xyz:
#         #
#         return X,Y,Z
#     else:
#         return alpha,beta,gamma



# Given dictionary of multipoles all with the same l, calculate the roated multipole with (l,mp)
def rotate_wfarrs_at_all_times( l,                          # the l of the new multipole (everything should have the same l)
                                m,                          # the m of the new multipole
                                like_l_multipoles_dict,     # dictionary in the format { (l,m): array([domain_values,+,x]) }
                                euler_alpha_beta_gamma,
                                ref_orientation = None ):             #

    '''
    Given dictionary of multipoles all with the same l, calculate the roated multipole with (l,mp).
    Key reference -- arxiv:1012:2879

    *NOTE* Note that the linear nature of this function allows it to be used for time OR frequency domain rotaitons.

    ~ LL,EZH 2018
    '''

    # Import usefuls
    from numpy import exp, pi, array, ones, sign, complex128
    from nrutils.manipulate.rotate import wdelement

    #
    alpha,beta,gamma = euler_alpha_beta_gamma

    #
    if not ( ref_orientation is None ) :
        error('The use of "ref_orientation" has been depreciated for this function.')

    # Handle the default behavior for the reference orientation
    if ref_orientation is None:
        ref_orientation = ones(3)

    # Apply the desired reflection for the reference orientation. NOTE that this is primarily useful for BAM run which have an atypical coordinate setup if Jz<0
    gamma *= sign( ref_orientation[-1] )
    alpha *= sign( ref_orientation[-1] )

    # Test to see if the original wfarr is complex and if so set the new wfarr to be complex as well
    wfarr_type = type( like_l_multipoles_dict[like_l_multipoles_dict.keys()[0]][:,1][0] )

    #
    # new_ylm = 0
    if wfarr_type == complex128:
        new_plus  = 0 + 0j
        new_cross = 0 + 0j
    else:
        new_plus  = 0
        new_cross = 0
    for lm in like_l_multipoles_dict:
        # See eq A9 of arxiv:1012:2879
        l,mp = lm
        old_wfarr = like_l_multipoles_dict[lm]

        #
        # y_mp = old_wfarr[:,1] + 1j*old_wfarr[:,2]
        # new_ylm += wdelement(l,m,mp,alpha,beta,gamma) * y_mp

        #
        d = wdelement(l,m,mp,alpha,beta,gamma)
        a,b = d.real,d.imag
        #
        p = old_wfarr[:,1]
        c = old_wfarr[:,2]
        #
        new_plus  += a*p - b*c
        new_cross += b*p + a*c

    # Construct the new waveform array
    t = old_wfarr[:,0]

    #
    # ans = array( [ t, new_ylm.real, new_ylm.imag ] ).T
    ans = array( [ t, new_plus, new_cross ] ).T

    # Return the answer
    return ans



# Careful function to find peak index for BBH waveform cases
def find_amp_peak_index( t, amp, phi, plot = False, return_jid=False ):

    '''
    Careful function to find peak index for BBH waveform cases
    e.g. when numerical junk is largeer than physical peak.
    '''

    #
    from numpy import log,argmax,linspace
    if plot:
        from matplotlib.pyplot import plot,yscale,xscale,axvline,show,figure,xlim

    # # defiune function for downsampling to speed up algo
    # def downsample(xx,yy,N):
    #     from positive import spline
    #     xx_ = spline( xx, xx )(linspace(xx[0],xx[-1],N))
    #     return xx_, spline(xx,yy)(xx_)
    #
    # # mask and downsample
    # mask =amp>0
    # tt = t[mask]
    # lamp = log( amp[mask] )
    # tt_,lamp_ = downsample(tt,lamp,300)
    #
    # # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Nknots = 6 # NOTE that this number is KEY to the algorithm: its related to the smalles number of lines needed to resolve two peaks! (4 lines) peak1 is junk peak2 is physical; if Nknots is too large, clustering of knots happens
    # # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    #
    # knots,_ = romline(tt_,lamp_,Nknots,positive=True,verbose=True)
    # if not (0 in knots):
    #     knots = [0] + [ knot for knot in knots ]
    #     Nknots += 1
    #
    # if plot:
    #     figure()
    #     axvline( tt_[knots[ int(Nknots/2) ]] )
    #     plot( tt, lamp )
    #     plot( tt_, lamp_ )
    #     plot( tt_[knots], lamp_[knots], color='r', lw=3, ls = '--' )
    #
    # #
    # pks,locs = findpeaks(lamp_[knots])
    #
    # # Clean amplitude if needed
    # if len(pks)>1: # if the peak is not the first knot == if this is not a ringdown only waveform
    #     refk = find( t>tt_[knots[ int(Nknots/2) ]] )[0]
    #     clean_amp = amp.copy()
    #     clean_amp[0:refk] = amp[ find( t>tt_[knots[ 0 ]] )[0] ]
    # else:
    #     # nothing need be done
    #     clean_amp = amp
    #
    # # Find peak index
    # k_amp_max = argmax( clean_amp )
    #
    # if plot:
    #     axvline( t[k_amp_max], color='k' )
    #     plot( t, log(clean_amp) )
    #     xlim( t[k_amp_max]-200,t[k_amp_max]+200 )
    #     show()

    amp_ = amp.copy()
    mask = amp_ > 1e-4*max(amp_)
    if sum(mask):
        a = find(mask)[0]
        b = find(mask)[-1]
        half_way = int((a+b)/2)
        amp_[ :half_way ] *= 0
        k_amp_max = argmax( amp_ )
        # handle ringdown cases
        if (k_amp_max == half_way): k_amp_max = 0

        pre = amp.copy()
        mask = pre > 1e-4*max(pre)
        a = find(mask)[0]
        b = find(mask)[-1]
        half_way = int((a+b)/2)
        pre[half_way:] *= 0
        dt = t[1]-t[0]
        jid = argmax( pre ) + int(100/dt)
        
        #
        if k_amp_max==0:
            
            #warning('the data input may not peak near merger. we will use the second derivative of the phase to estimate the location of merger in the timeseries')
        
            mask = amp>(1e-2*max(amp))
            d2phi = abs(spline_diff(t[mask],phi[mask],n=2))
            
            k_amp_max = argmax(d2phi)
            ans = k_amp_max
            jid=0

        # Return answer
        ans = k_amp_max
        if return_jid: ans = (k_amp_max,jid)
        return ans
    else:
        
        # print sum(amp),sum(phi),sum(amp)<1e-6
        if sum(phi)<1e-6:
            ans = 0
            if return_jid: ans = (k_amp_max,0)
            warning('the data appears to be ZERO; please make sure you understand why')
            return ans
        
        warning('the data input here may be flat.')
        
        k_amp_max = 0
        jid=0
        ans = k_amp_max
        if return_jid: ans = (0,0)
        return ans
