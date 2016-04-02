
# Import core basics
from nrutils.core.basics import *

# --------------------------------------------------------------- #
# Physical Constants
# --------------------------------------------------------------- #
__physical_constants__ = {  'mass_sun'          : 1.98892e30,		# kg
                            'G'                 : 6.67428e-11,		# m^3/(kg s^2)
                            'c'                 : 2.99792458e8,		# m/s
                            'meter_to_mpc'      : 3.24077649e-23}	# meter to mpc conversion


# mass of the sun in secs
__physical_constants__['mass_sun_secs'] = __physical_constants__['G']*__physical_constants__['mass_sun']/(__physical_constants__['c']*__physical_constants__['c']*__physical_constants__['c'])
# mass of the sun in meters
__physical_constants__['mass_sun_meters'] = __physical_constants__['G']*__physical_constants__['mass_sun']/(__physical_constants__['c']*__physical_constants__['c'])
# mass of the sun in Mpc
__physical_constants__['mass_sun_mpc'] = __physical_constants__['mass_sun_meters'] * __physical_constants__['meter_to_mpc']

# --------------------------------------------------------------- #
# Given TIME DOMAIN strain in physical units, convert to Code units
# --------------------------------------------------------------- #
def codeh( harr, M, D ):

    # convert time series to physical units
    harr[:,0] = codet( harr[:,0], M )

    # scale wave amplitude for mass and distance
    harr[:,1:] =  harr[:,1:] / (mass_mpc( M )/D)

    #
    return harr

# --------------------------------------------------------------- #
# Given FREQUENCY DOMAIN strain in physical units, convert to Code units
# --------------------------------------------------------------- #
def codehfd( fd_harr, M, D ):

    # convert time series to physical units
    fd_harr[:,0] = codef( fd_harr[:,0], M )

    # scale wave amplitude for mass and distance
    fd_harr[:,1:] =  fd_harr[:,1:] / (mass_mpc( M )/D)
    # convert the integration factor, dt, to code units
    fd_harr[:,1:] =  fd_harr[:,1:] / mass_sec(M)

    #
    return fd_harr

# --------------------------------------------------------------- #
# Convert physical time to code units
# --------------------------------------------------------------- #
def codet( t, M ): return t/mass_sec(M)

# --------------------------------------------------------------- #
# Convert physical frequency series to code units
# --------------------------------------------------------------- #
def codef( f, M ): return f*mass_sec(M)

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
