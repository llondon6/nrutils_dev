
#
from nrutils.core.basics import *
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt,inf,nan,sign
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None, verbose = True ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join,basename
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # The folder is valid if there is l=m=2 mode data in the following dirs
    status = len( ls( run_dir + '/rStrainByMass2_l2_m1*' ) ) > 0
    status = status or len( ls( run_dir + '/rStrainByMass2_l2_m2*' ) ) > 0

    # Let the people know
    if not status:
        msg = 'waveform data could not be found.'
        if verbose: warning(msg,'maya.validate')

    # ignore directories with certain tags in filename
    ignore_tags = ['backup','old','archive']
    for tag in ignore_tags:
        status = status and not ( tag in run_dir )

    # ensure that file name is the same as the folder name
    a = basename(metadata_file_location).split(config.metadata_id)[0]
    b = parent(metadata_file_location)
    status = status and (  a in b  )

    #
    return status


#
def learn_metadata( metadata_file_location ):

    # Learn the par file
    raw_metadata = smart_object( metadata_file_location )

    # Shortand
    y = raw_metadata

    # Create smart_object for the standard metadata
    standard_metadata = smart_object()
    # shorthand
    x = standard_metadata

    # Creation date of metadata file
    x.date_number = getctime(  metadata_file_location  )

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate derivative quantities  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Masses
    x.m1 = 1-1e-5
    x.m2 = 1e-5

    #
    P1 = array( [ 1,1,0 ] )
    P2 = array( [ 1,1,0 ] )

    #
    S1 = array( [ 0,0,y.a ] )
    S2 = array( [ 0,0,0 ] )

    # Read initial locations from the ShiftTracker files
    R1 = array( [ 0,0,0 ] )
    R2 = array( [ 12.0,0,0 ] )

    # Find initial binary separation for convenience
    x.b = norm(R1-R2)

    #
    x.note = 'Binary parameters correspond to initial data, not an after-junk point.'

    # Estimate the component angular momenta
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    # Extract and store the initial adm energy
    x.madm = 1.0

    # Store the initial linear momenta
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2)

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    #
    L = L1+L2
    S = S1+S2
    x.L = L
    x.J = L+S
    
    # Store related final mass and spin data
    x.mf = 1.0
    x.Sf = array([y.a])
    x.Xf = array([0,0,])
    x.xf = sign(x.Sf[-1])*norm(x.Sf)/(x.mf*x.mf)

    # Store relaxed (after-junk) fields
    x.S1_afterjunk,x.S_afterjunk2,x.S_afterjunk = None,None,None
    x.L1_afterjunk,x.L2_afterjunk,x.L_afterjunk = None,None,None
    x.R1_afterjunk,x.R2_afterjunk = None,None
    x.P1_afterjunk,x.P2_afterjunk = None,None
    x.J_afterjunk = None
    
    #
    x.S = x.S1 + x.S2
    x.L = x.L1 + x.L2
    x.J = x.L + x.S

    #
    x.valid = True

    #
    return standard_metadata, raw_metadata


# Given an extraction parameter, return an extraction radius
def extraction_map( this, extraction_parameter, verbose=False ):

    # Given an extraction parameter, return an extraction radius
    extraction_radius = extraction_parameter

    #
    return extraction_radius


# Estimate a good extraction radius and level for an input scentry object from the BAM catalog
def infer_default_level_and_extraction_parameter( this,     # An scentry object
                                                  desired_exraction_radius=None,    # (Optional) The desired extraction radius in M, where M is the initial ADM mass
                                                  verbose=None ):   # Toggel to let the people know
    '''Estimate a good extraction radius and level for an input scentry object from the BAM catalog'''

    # NOTE that input must be scentry object
    # Import useful things
    from glob import glob
    from numpy import array,argmin,inf
    
    #
    extraction_parameter = inf

    # Also store a dictionary between extraction parameter and extraction radius
    # And a dictionary between the level parameter and extraction radius.
    # For Maya this is not used and so it's set to None. BAM uses it though.
    extraction_map_dict = {}
    extraction_map_dict['radius_map'] = { inf:inf }
    extraction_map_dict['level_map'] = None

    # Note that maya sims have no level specification
    level = None

    # Return answers
    return extraction_parameter,level,extraction_map_dict
