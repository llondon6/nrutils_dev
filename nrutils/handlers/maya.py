
#
from nrutils.core.basics import *
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt,inf,nan
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join,basename
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # The folder is valid if there is l=m=2 mode data in the following dirs
    status = len( ls( run_dir + '/Ylm_WEYLSCAL4::Psi4r_l2_m1_r*' ) ) > 0
    status = status or len( ls( run_dir + '/mp_WeylScal4::Psi4i_l2_m2_r*' ) ) > 0

    # Let the people know
    if not status:
        msg = 'waveform data could not be found.'
        warning(msg,'maya.validate')

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

    #
    thisfun = 'maya.learn_metadata'

    # Look for stdout files
    stdout_file_list = ls( parent(metadata_file_location)+'/stdout*' )
    if not stdout_file_list:
        msg = 'cannot find stdout files which contain important metadata'
        error(msg,'maya.learn_metadata')
    # Look for ShiftTracker files
    shift_tracker_file_list = ls( parent(metadata_file_location)+'/Shift*' )
    if not shift_tracker_file_list:
        msg = 'cannot find ShiftTracker* files which contain information about binary dynamics'
        error(msg,'maya.learn_metadata')
    # Look for Horizon mass and spin files
    hn_file_list = ls( parent(metadata_file_location)+'/hn*' )
    if not hn_file_list:
        msg = 'cannot find hn_masspin files which contain information about remnant BH final state'
        error(msg,'maya.learn_metadata')

    # Use the first file returned by the OS
    # NOTE that this file is neeeded to get the component and net ADM masses
    stdout_file_location = stdout_file_list[0]

    # Learn the par file
    raw_metadata = smart_object( metadata_file_location )

    # Shortand
    y = raw_metadata

    # Retrieve ADM masses form the stdout file
    y.learn_string( grep( 'm+', stdout_file_location )[0] )
    y.learn_string( grep( 'm-', stdout_file_location )[0] )
    y.learn_string( grep( 'ADM mass from r', stdout_file_location )[0] )

    # # Useful for debugging -- show what's in y
    # y.show()

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
    x.m1 = getattr(y,'INFO(TwoPunctures):ADMmassforpuncture:m_')
    x.m2 = getattr(y,'INFO(TwoPunctures):ADMmassforpuncture:m+')

    #
    P1 = array( [ getattr(y,'twopunctures::par_P_plus[0]'),
                  getattr(y,'twopunctures::par_P_plus[1]'),
                  getattr(y,'twopunctures::par_P_plus[2]') ] )
    P2 = array( [ getattr(y,'twopunctures::par_P_minus[0]'),
                  getattr(y,'twopunctures::par_P_minus[1]'),
                  getattr(y,'twopunctures::par_P_minus[2]') ] )

    #
    S1 = array( [ getattr(y,'twopunctures::par_s_plus[0]'),
                  getattr(y,'twopunctures::par_s_plus[1]'),
                  getattr(y,'twopunctures::par_s_plus[2]') ] )
    S2 = array( [ getattr(y,'twopunctures::par_s_minus[0]'),
                  getattr(y,'twopunctures::par_s_minus[1]'),
                  getattr(y,'twopunctures::par_s_minus[2]') ] )

    # Read initial locations from the ShiftTracker files
    def shiftt_to_initial_bh_location(key):
        shiftt0_file_location = [ f for f in shift_tracker_file_list if key in f ][0]
        fid = open( shiftt0_file_location )
        return array( [ float(a) for a in fid.readline().replace('\n','').split('\t')][2:5] )
    R1 = shiftt_to_initial_bh_location("ShiftTracker0")
    R2 = shiftt_to_initial_bh_location("ShiftTracker1")

    # Find initial binary separation for convenience
    x.b = norm(R1-R2)

    #
    x.note = 'Binary parameters correspond to initial data, not an after-junk point.'

    # Estimate the component angular momenta
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    # Extract and store the initial adm energy
    x.madm = getattr(y,'INFO(TwoPunctures):ADMmassfromr')[-1]

    # Store the initial linear momenta
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2)

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    # Load Final Mass and Spin Data  hn_mass_spin_2
    hn_file_bin = [ f for f in hn_file_list if 'hn_mass_spin_2' in f ]
    proceed = len(hn_file_bin)==1
    hn_file = hn_file_bin[0] if proceed else None
    nan_remnant_data = array( [ nan,nan,nan,nan,nan ] )

    #
    if not proceed:
        #
        msg = 'The default hn_mass_spin_2 file could not be found. Place-holders (i.e. nans) for the remnant information will be passed.'
        warning(msg,thisfun)
        x.note += ' ' + msg
        remnant_data = nan_remnant_data
    else:
        # Use bash's tail to get the last row in the file
        cmd = 'tail -n 1 %s' % hn_file
        data_string = bash(cmd)
        # If the string is empty, then there was an error
        if not data_string:
            msg = 'The system failed using tail to get the remnant state from \"%s\"'%cyan(cmd)
            error(msg,thisfun)
        # Else, parse the data string into floats
        remnant_data = [ float(v) for v in data_string.replace('\n','').split('\t') ]
        # Handle formatting cases
        if len(remnant_data) != 5:
            msg = 'Remanant data was loaded, but its format is unexpected (last row length is %s). Placeholders for the remnant information will be passed.' % yellow( str(len(remnant_data)) )
            warning(msg,thisfun)
            x.note += ' ' + msg
            remnant_data = nan_remnant_data

    # Unpack the remnant data
    [tf,Mf,xfx,xfy,xfz] = remnant_data
    # Store related final mass and spin data
    x.mf = Mf
    x.xf = array([xfx,xfy,xfz])
    x.Sf = Mf*Mf*x.xf

    # True if extraction parameter is extraction radius
    x.extraction_parameter_is_radius = True

    #
    x.valid = True

    #
    return standard_metadata, raw_metadata
