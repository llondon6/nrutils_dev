
# Import core basics
from nrutils.core.basics import *

# --------------------------------------------------------------- #
# Physical Constants
# --------------------------------------------------------------- #
__physical_constants__ = {  'mass_sun'          : 1.98892e30,		# kg
                            'G'                 : 6.67428e-11,		# m^3/(kg s^2)
                            'c'                 : 2.99792458e8,		# m/s
                            'meter_to_mpc'      : 3.24077649e-23}	# meter to Mpc conversion


# mass of the sun in secs
__physical_constants__['mass_sun_secs'] = __physical_constants__['G']*__physical_constants__['mass_sun']/(__physical_constants__['c']*__physical_constants__['c']*__physical_constants__['c'])
# mass of the sun in meters
__physical_constants__['mass_sun_meters'] = __physical_constants__['G']*__physical_constants__['mass_sun']/(__physical_constants__['c']*__physical_constants__['c'])
# mass of the sun in Mpc
__physical_constants__['mass_sun_mpc'] = __physical_constants__['mass_sun_meters'] * __physical_constants__['meter_to_mpc']


# --------------------------------------------------------------- #
# Given FREQUENCY DOMAIN strain in code units, convert to Physical units
# --------------------------------------------------------------- #
def physhf( harr, M, D ):
    '''
    Given FREQUENCY DOMAIN strain in code units, convert to Physical units

    code_hf_array = physhf( geometric_hf_array, M_solar, D_Mpc )

    '''
    # Import useful things
    from numpy import ndarray
    # Calculate unit conversion factor for strain amplitude
    K =  mass_mpc(M)/D  # Scale according to total mass and distance
    K *= mass_sec(M)    # Scale the fourier transform's dt to physical units
    # If conversion of a number is desired
    if isinstance(harr,(float,int,complex,ndarray)):
        # Convert strain data to physical units and return
        return harr*K
    elif isinstance(harr,list):
        # Convert strain data to physical units and return
        return [ K*hj for hj in harr ]
    elif harr.__class__.__name__=='gwf':
        # Convert gwf to physical units
        y = harr.copy()
        phys_wfarr = physh( y.wfarr, M, D )
        from nrutils import gwf
        return gwf( phys_wfarr )
    else:
        # Here we will asusme that input is numpy ndarray
        if 3 == harr.shape[-1]:
            # Convert the frequency column to physical units
            harr[:,0] = physf( harr[:,0], M )
            # Convert strain data to physical units
            harr[:,1:] *= K
            # Return answer
            return harr
        elif 1==len(harr.shape):
            harr *= K

# --------------------------------------------------------------- #
# Given TIME DOMAIN strain in code units, convert to Physical units
# --------------------------------------------------------------- #
def physh( harr, M, D ):
    '''Given TIME DOMAIN strain (waveform array OR number) in code units, convert to Physical units'''
    # Calculate unit conversion factor for strain amplitude
    K = mass_mpc(M)/D
    # If conversion of a number is desired
    if isinstance(harr,(float,int,complex)):
        # Convert strain data to physical units and return
        return harr*K
    elif isinstance(harr,list):
        # Convert strain data to physical units and return
        return [ K*hj for hj in harr ]
    else:
        # Here we will asusme that input is numpy ndarray
        if 3 == harr.shape[-1]:
            # Convert the time column to physical units
            harr[:,0] = physt( harr[:,0], M )
            # Convert strain data to physical units
            harr[:,1:] *= K
            # Return answer
            return harr
        elif 1==len(harr.shape):
            harr *= K

# --------------------------------------------------------------- #
# Given TIME DOMAIN strain in physical units, convert to Code units
# --------------------------------------------------------------- #
def codeh( harr, M, D ):
    '''Given TIME DOMAIN strain in physical units, convert to Code units'''
    # convert time series to physical units
    harr[:,0] = codet( harr[:,0], M )

    # scale wave amplitude for mass and distance
    harr[:,1:] =  harr[:,1:] / (mass_mpc( M )/D)

    #
    return harr

# --------------------------------------------------------------- #
# Given FREQUENCY DOMAIN strain in physical units, convert to Code units
# --------------------------------------------------------------- #
def codehf( fd_harr, M, D ):
    '''Given FREQUENCY DOMAIN strain in physical units, convert to Code units'''
    #
    K  = D/mass_mpc( M )    # scale wave amplitude for mass and distance
    K /= mass_sec(M)        # convert the differential dt to M
    if fd_harr.shape[-1] == 3 :
        # convert freq series to physical units
        fd_harr[:,0] = codef( fd_harr[:,0], M )
        # scale wave amplitude for mass and distance and convert the differential dt to M
        fd_harr[:,1:] =  fd_harr[:,1:] * K
    elif len(fd_harr.shape)==1:
        fd_harr *= K

    #
    return fd_harr

# --------------------------------------------------------------- #
# Convert physical time to code units
# --------------------------------------------------------------- #
def codet( t, M ):
    '''Convert physical time (sec) to code units'''
    return t/mass_sec(M)

# --------------------------------------------------------------- #
# Convert physical frequency series to code units
# --------------------------------------------------------------- #
def codef( f, M ):
    '''Convert physical frequency series (Hz) to code units'''
    return f*mass_sec(M)

# --------------------------------------------------------------- #
# Convert code frequency to physical frequency
# --------------------------------------------------------------- #
def physf( f, M ):
    '''Convert code frequency to physical frequency (Hz)'''
    from numpy import ndarray
    if isinstance(f,(tuple,list) ):
        return [ ff/mass_sec(M) for ff in f ] if isinstance(f,list) else ( ff/mass_sec(M) for ff in f )
    else:
        return f/mass_sec(M)

# --------------------------------------------------------------- #
# Convert code time to physical time (sec)
# --------------------------------------------------------------- #
def physt( t, M ):
    '''Convert code time to physical time (sec)'''
    return t*mass_sec(M)

# --------------------------------------------------------------- #
# Convert mass in code units to seconds
# --------------------------------------------------------------- #
def mass_sec( M ): return M*__physical_constants__['mass_sun_secs']

# --------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------- #
def mass_mpc( M ): return M*__physical_constants__['mass_sun_mpc']


# Convert component masses to mass ratio
def m1m2q(m1,m2): return float(max([m1,m2]))/min([m1,m2])

# Convert q to eta
def q2eta(q): return q/((1.0+q)*(1.0+q))

# Convert eta to q
def eta2q(eta):
    from numpy import sqrt
    if eta>0.25:
        raise ValueError('eta must be less than 0.25, but %f found'%eta)
    b = 2.0 - 1.0/eta
    q_plus  = (-b + sqrt( b*b - 4 ))/2 # m1/m2
    q_minus = (-b - sqrt( b*b - 4 ))/2 # m2/m1
    if q_plus.imag:
        warning( 'eta = %1.4f> 0.25, eta must be <= 0.25'%eta )
    return q_plus
