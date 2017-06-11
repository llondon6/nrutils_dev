#
from positive import *


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
def intrp_wfarr(wfarr,delta=None,domain=None):

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
        d = wfarr[0,0]-wfarr[1,0]
        if abs(delta-d)/delta < 1e-6:
            proceed = False

    # If there is need to interpolate, then interpolate.
    if proceed:

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

        # Interpolate the remaining columns
        for k in range(1,wfarr.shape[1]):
            _wfarr[:,k] = spline( input_domain, wfarr[:,k] )( intrp_domain )

    else:

        # Otherwise, return the input array
        _wfarr = wfarr

    #
    return _wfarr


# Fucntion to pad wfarr with zeros. NOTE that this should only be applied to a time domain waveform that already begins and ends with zeros.
def pad_wfarr(wfarr,new_length,where=None):

    #
    from numpy import hstack,zeros,arange

    # Only pad if size of the array is to increase
    length = len(wfarr[:,0])
    proceed = length < new_length

    #
    if isinstance(where,str):
        where = where.lower()

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

        if where is 'sides':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length)/2 )
            right_pad = zeros( new_length-length-len(left_pad) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k],right_pad] )

        elif where == 'right':

            # Create the pads for the other columns
            right_pad = zeros( new_length-length )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [wfarr[:,k],right_pad] )

        elif where == 'left':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k]] )

    else:

        # Otherwise, do nothing.
        _wfarr = wfarr

        # Warn the user that nothing has happened.
        msg = 'The desired new length is <= the current array length (i.e. number of time domain points). Nothing will be padded.'
        warning( msg,fname='pad_wfarr'+cyan('@%i'%linenum()) )

    # Return padded array
    return _wfarr

# Shift a waveform arra by some "shift" amount in time
def tshift_wfarr( _wfarr, shift ):
    '''Shift a waveform arra by some "shift" amount in time'''
    # Import useful things
    from numpy import array
    # Unpack waveform array
    t,p,c = _wfarr[:,0],_wfarr[:,1],_wfarr[:,2]
    _y = p + 1j*c
    # Shift the waveform array data using tshift
    y = tshift( t,_y,shift )
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
    from numpy import angle,unwrap,mean

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
    from numpy import ndarray,zeros,dot,array

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
        Z = dot( M,Y )[:,0]
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
def get_wfarr_relative_phase(this,that):

    #
    from numpy import angle,unwrap,mean

    #
    u = this[:,1]+1j*this[:,2]
    v = that[:,1]+1j*that[:,2]

    #
    _a = unwrap( angle(u) )[0]
    _b = unwrap( angle(v) )[0]

    #
    dphi = -_a + _b

    #
    return dphi


# Find the average phase difference and align two wfarr's
def align_wfarr_initial_phase(this,that):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    dphi = get_wfarr_relative_phase(this,that)

    #
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
    from numpy import arange,sum,array,diff,isfinite,hstack
    thisfun = 'straighten_wfarr'

    # Remove rows that contain non-finite data
    finite_mask = isfinite( sum( wfarr, 1 ) )
    if sum(finite_mask)!=len(finite_mask):
        if verbose: alert('Non-finite values found in waveform array. Corresponding rows will be removed.',thisfun)
    wfarr = wfarr[ finite_mask, : ]

    # Sort rows by the time series' values
    time = array( wfarr[:,0] )
    space = arange( wfarr.shape[0] )
    chart = sorted( space, key = lambda k: time[k] )
    if (space != chart).all():
        if verbose: alert('The waveform array was found to have nonmonotinicities in its time series. The array will now be straightened.',thisfun)
    wfarr = wfarr[ chart, : ]

    # Remove rows with duplicate time series values
    time = array( wfarr[:,0] )
    diff_mask = hstack( [ True, diff(time).astype(bool) ] )
    if sum(diff_mask)!=len(diff_mask):
        if verbose: alert('Repeated time values were found in the array. Offending rows will be removed.',thisfun)
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
    if verbose: print dpsi_opt,action([dphi,dpsi_opt])
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

    if verbose:
        print 'dphi_opt = ' + str(dphi_opt)
        print 'dpsi_opt = ' + str(dpsi_opt)
        print 'val_max = ' + str(val_max)

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
        print '>> ',dphi,dpsi,action([dphi,dpsi])
        k+=1

    return dphi,dpsi
