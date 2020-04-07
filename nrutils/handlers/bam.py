
#
from nrutils.core.basics import *
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt,sign
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None, verbose = False ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join,basename,getsize
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # The folder is valid if there is l=m=2 mode data in the following dirs
    min_file_list = ls( run_dir + '/Psi4ModeDecomp/psi3col*l2.m2.gz' )
    status = len( min_file_list ) > 0
    if not status: warning('Cannot find the (2,2) multipole for %s. This will cause the simulation to be marked as invalid.'%yellow(metadata_file_location))

    # Ensuer that data is non-empty
    status = getsize( min_file_list[0] ) > 25 if status else False
    if not status: warning('The (2,2) multipole for %s looks empty. This will cause the simulation to be marked as invalid.'%yellow(metadata_file_location))

    # ignore directories with certain tags in filename
    ignore_tags = ['backup','old']
    for tag in ignore_tags:
        status = status and not ( tag in run_dir )

    #
    a = basename(metadata_file_location).split(config.metadata_id)[0]
    b = parent(metadata_file_location)
    status = status and (  a in b  )

    #
    return status

# Learn the metadta (file) for this type of NR waveform
# NOTE that this version adds separate fields for relaxed and non-relaxed quantities
def learn_metadata( metadata_file_location ):

    # Try to load the related par file as well as the metadata file
    from glob import glob as find
    from os.path import dirname

    ## NOTE that some people (I mean you Chinmay!) don't always use the run name
    ## for their par files, so here we will find and use the first par file reported
    ## by the OS. NOTE that the approach commented out below is prefered.
    # par_file_location = metadata_file_location[:-3]+'par'

    par_file_location = find(dirname(metadata_file_location)+'/*.par')[0]
    raw_metadata = smart_object( [metadata_file_location,par_file_location] )

    # shortand
    y = raw_metadata

    # # Useful for debugging -- show what's in y
    # y.show()

    #
    standard_metadata = smart_object()
    # shorthand
    x = standard_metadata

    # Keep NOTE of important information
    x.note = ''

    # Creation date of metadata file
    x.date_number = getctime(  metadata_file_location  )

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate derivative quantities  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Masses
    x.m1 = y.mass1
    x.m2 = y.mass2

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
    # Define/Store ADM initial parameters
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
    # Spins
    S1 = array( [ y.initial_bh_spin1x, y.initial_bh_spin1y, y.initial_bh_spin1z ] )
    S2 = array( [ y.initial_bh_spin2x, y.initial_bh_spin2y, y.initial_bh_spin2z ] )
    S = S1 + S2
    # Momenta
    P1 = array( [ y.initial_bh_momentum1x, y.initial_bh_momentum1y, y.initial_bh_momentum1z ] )
    P2 = array( [ y.initial_bh_momentum2x, y.initial_bh_momentum2y, y.initial_bh_momentum2z ] )
    # positions
    R1 = array( [ y.initial_bh_position1x, y.initial_bh_position1y, y.initial_bh_position1z ] )
    R2 = array( [ y.initial_bh_position2x, y.initial_bh_position2y, y.initial_bh_position2z ] )
    # Estimate the component angular momenta
    try:
        L1 = cross(R1,P1)
        L2 = cross(R2,P2)
    except:
        error('There was an insurmountable problem encountered when trying to load initial binary configuration. For example, %s. The guy at the soup shop says "No soup for you!!"'%red('P1 = '+str(P1)))
    # orbital angular momentum
    L = L1+L2
    L_ = array( [ y.initial_angular_momentumx, y.initial_angular_momentumy, y.initial_angular_momentumz ] )
    if norm((L-L_)/norm(0.5*(L+L_))) > 1e-1:
        msg = cyan('Warning:')+' The estimated L differs from that in the bbh file by more than 10\%'+'. The user should understand whether this is an error or not.'
        x.note += msg
        warning(msg,'bam.py')
    # Total angular momentum
    J = L+S
    # Store all quantities to the current object
    x.S1,x.S2,x.S = S1,S2,S
    x.L1,x.L2,x.L = L1,L2,L
    x.R1,x.R2 = R1,R2
    x.P1,x.P2 = P1,P2
    x.J = J

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
    # Define/Store relaxed (a.k.a after-junk) initial parameters
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#

    # NOTE that some bbh files may not have after_junkradiation_spin data (i.e. empty). In these cases we will take the initial spin data
    S1 = array( [ y.after_junkradiation_spin1x, y.after_junkradiation_spin1y, y.after_junkradiation_spin1z ] )
    S2 = array( [ y.after_junkradiation_spin2x, y.after_junkradiation_spin2y, y.after_junkradiation_spin2z ] )

    #%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%#
    # NOTE that sometimes the afterjunk spins may not be stored correctely or at all in the bbh files. Therefore an additional validation step is needed here.
    # -- NOTE the lines below need to be fixed --
    S1bool = S1.astype(list).astype(bool)
    S2bool = S2.astype(list).astype(bool)
    x.has_valid_relaxed_intial_parameters = S1bool.all() and S2bool.all()
    #%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%#

    # If the data is to be stored using afterjunk parameters:
    if x.has_valid_relaxed_intial_parameters:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Use afterjunk information                   #
        msg = cyan('Initial parameters corresponding to the bbh file\'s aftrejunktime will be used to populate metadata.')
        alert(msg,'bam.py')
        x.note += msg
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # total spin
        S = S1+S2
        # find puncture data locations
        puncture_data_1_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate1*' )[0]
        puncture_data_2_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate2*' )[0]

        # load puncture data
        puncture_data_1,_ = smart_load( puncture_data_1_location )
        puncture_data_2,_ = smart_load( puncture_data_2_location )

        # Mask away the initial junk region using the after-junk time given in the bbh metadata
        after_junkradiation_time = y.after_junkradiation_time
        after_junkradiation_mask = puncture_data_1[:,-1] > after_junkradiation_time

        puncture_data_1 = puncture_data_1[ after_junkradiation_mask, : ]
        puncture_data_2 = puncture_data_2[ after_junkradiation_mask, : ]

        R1 = array( [  puncture_data_1[0,0],puncture_data_1[0,1],puncture_data_1[0,2],  ] )
        R2 = array( [  puncture_data_2[0,0],puncture_data_2[0,1],puncture_data_2[0,2],  ] )

        # NOTE that here the shift is actually contained within puncture_data, and NOTE that the shift is -1 times the velocity
        P1 = x.m1 * -array( [  puncture_data_1[0,3],puncture_data_1[0,4],puncture_data_1[0,5],  ] )
        P2 = x.m2 * -array( [  puncture_data_2[0,3],puncture_data_2[0,4],puncture_data_2[0,5],  ] )

        # Estimate the component angular momenta
        try:
            L1 = cross(R1,P1)
            L2 = cross(R2,P2)
            L = L1+L2
        except:
            error('There was an insurmountable problem encountered when trying to load initial binary configuration. For example, %s. The guy at the soup shop says "No soup for you!!"'%red('P1 = '+str(P1)))

        # Total angular momentum
        J = L+S

    else:

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Use initial data information                #
        msg = cyan('Warning:')+yellow(' The afterjunk spins appear to have been stored incorrectly in the bbh file. Values of NONE will be stored to the current object. ')
        warning(msg,'bam.py')
        x.note += msg
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # NOTE that these parameters were defined above, and so do not need to be redefined here.
        # NOTE that the lines below are commented out resulting in the afterjunk quantities being the initial quantities in cases where the data are not properly formatted. This is confusing. I'm sorry.
        #S1,S2,S = None,None,None
        #L1,L2,L = None,None,None
        #R1,R2 = None,None
        #P1,P2 = None,None
        #J = None

    # Extract and store the initial adm energy
    x.madm = y.initial_ADM_energy

    # Store all quantities to the current object
    # --

    # NOTE that previously aftejunk fields had been reatined for backwards compatability. Now they are depreciated and hence commented out below. IN THE FUTURE, a separate methid may be developed to carefully store these time/freq dependednt (by convention of refernce) parameters.
    ## x.S1_afterjunk,x.S2_afterjunk,x.S_afterjunk = S1,S2,S
    ## x.L1_afterjunk,x.L2_afterjunk,x.L_afterjunk = L1,L2,L
    ## x.R1_afterjunk,x.R2_afterjunk = R1,R2
    ## x.P1_afterjunk,x.P2_afterjunk = P1,P2
    ## x.J_afterjunk = J

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2) # float( y.initial_separation )
    if abs( y.initial_separation - norm(R1-R2) ) > 1e-1:
        msg = cyan('Warning:')+' The estimated after junk binary separation is significantly different than the value stored in the bbh file: '+yellow('x from calculation = %f, x from bbh file=%f' % (norm(R1-R2),y.initial_separation) )+'. The user should understand whether this is an error or not.'
        x.note += msg
        warning(msg,'bam.py')
    # Let the use know that the binary separation is possibly bad
    if x.b<4:
        msg = cyan('Warning:')+' The estimated initial binary separation is very small. This may be due to an error in the puncture data. You may wish to use the initial binary separation from the bbh file which is %f'%y.initial_separation+'. '
        warning(msg,'bam.py')
        x.note += msg

    #
    x.valid = True

    # Load irriducible mass data
    mass_file_list = ls(parent(metadata_file_location)+'hmass_2*gz')
    if len(mass_file_list)>0:
        mass_file = mass_file_list[0]
        mass_data,mass_status = smart_load(mass_file)
    else:
        mass_status = False
    # Load spin data
    spin_file_list = ls(parent(metadata_file_location)+'hspin_2*gz')
    if len(spin_file_list)>0:
        spin_file = spin_file_list[0]
        spin_data,spin_status = smart_load(spin_file)
    else:
        spin_status = False
    # Estimate final mass and spin
    if mass_status and spin_status:
        Sf = spin_data[-1,1:]
        Mf = mass_data[-1,1]
        x.__Mf__ = Mf
        x.spin_data_series = spin_data
        x.mass_data_series = mass_data
        x.mf = Mf
        #
        x.Sf = Sf
        x.Xf = x.Sf/(x.mf*x.mf)
        x.xf = sign(x.Sf[-1])*norm(x.Sf)/(x.mf*x.mf)
    else:
        from numpy import nan
        x.Sf = nan*array([0.0,0.0,0.0])
        x.Xf = nan*array([0.0,0.0,0.0])
        x.mf = nan
        x.xf = nan

    #
    return standard_metadata, raw_metadata


# Learn the metadta (file) for this type of NR waveform
def learn_metadata_legacy( metadata_file_location ):

    # Try to load the related par file as well as the metadata file
    from glob import glob as find
    from os.path import dirname

    ## NOTE that some people (I mean you Chinmay!) don't always use the run name
    ## for their par files, so here we will find and use the first par file reported
    ## by the OS. NOTE that the approach commented out below is prefered.
    # par_file_location = metadata_file_location[:-3]+'par'

    par_file_location = find(dirname(metadata_file_location)+'/*.par')[0]
    raw_metadata = smart_object( [metadata_file_location,par_file_location] )

    # shortand
    y = raw_metadata

    # # Useful for debugging -- show what's in y
    # y.show()

    #
    standard_metadata = smart_object()
    # shorthand
    x = standard_metadata

    # Keep NOTE of important information
    x.note = ''

    # Creation date of metadata file
    x.date_number = getctime(  metadata_file_location  )

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate derivative quantities  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Masses
    x.m1 = y.mass1
    x.m2 = y.mass2

    # NOTE that some bbh files may not have after_junkradiation_spin data (i.e. empty). In these cases we will take the initial spin data
    S1 = array( [ y.after_junkradiation_spin1x, y.after_junkradiation_spin1y, y.after_junkradiation_spin1z ] )
    S2 = array( [ y.after_junkradiation_spin2x, y.after_junkradiation_spin2y, y.after_junkradiation_spin2z ] )

    #%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%#
    # NOTE that sometimes the afterjunk spins may not be stored correctely or at all in the bbh files. Therefore an additional validation step is needed here.
    S1bool = S1.astype(list).astype(bool)
    S2bool = S2.astype(list).astype(bool)
    x.isafterjunk = S1bool.all() and S2bool.all()
    #%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%@%%#

    # If the data is to be stored using afterjunk parameters:
    if x.isafterjunk:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Use afterjunk information                   #
        msg = cyan('Initial parameters corresponding to the bbh file\'s aftrejunktime will be used to populate metadata.')
        alert(msg,'bam.py')
        x.note += msg
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # find puncture data locations
        puncture_data_1_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate1*' )[0]
        puncture_data_2_location = ls( parent( metadata_file_location )+ 'moving_puncture_integrate2*' )[0]

        # load puncture data
        puncture_data_1,_ = smart_load( puncture_data_1_location )
        puncture_data_2,_ = smart_load( puncture_data_2_location )

        # Mask away the initial junk region using the after-junk time given in the bbh metadata
        after_junkradiation_time = y.after_junkradiation_time
        after_junkradiation_mask = puncture_data_1[:,-1] > after_junkradiation_time

        puncture_data_1 = puncture_data_1[ after_junkradiation_mask, : ]
        puncture_data_2 = puncture_data_2[ after_junkradiation_mask, : ]

        R1 = array( [  puncture_data_1[0,0],puncture_data_1[0,1],puncture_data_1[0,2],  ] )
        R2 = array( [  puncture_data_2[0,0],puncture_data_2[0,1],puncture_data_2[0,2],  ] )

        # NOTE that here the shift is actually contained within puncture_data, and NOTE that the shift is -1 times the velocity
        P1 = x.m1 * -array( [  puncture_data_1[0,3],puncture_data_1[0,4],puncture_data_1[0,5],  ] )
        P2 = x.m2 * -array( [  puncture_data_2[0,3],puncture_data_2[0,4],puncture_data_2[0,5],  ] )
    else:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Use initial data information                #
        msg = cyan('Warning:')+yellow(' The afterjunk spins appear to have been stored incorrectly. All parameters according to the initial data (as stored in the bbh files) will be stored. ')
        warning(msg,'bam.py')
        x.note += msg
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Spins
        S1 = array( [ y.initial_bh_spin1x, y.initial_bh_spin1y, y.initial_bh_spin1z ] )
        S2 = array( [ y.initial_bh_spin2x, y.initial_bh_spin2y, y.initial_bh_spin2z ] )
        # Momenta
        P1 = array( [ y.initial_bh_momentum1x, y.initial_bh_momentum1y, y.initial_bh_momentum1z ] )
        P2 = array( [ y.initial_bh_momentum2x, y.initial_bh_momentum2y, y.initial_bh_momentum2z ] )
        # positions
        R1 = array( [ y.initial_bh_position1x, y.initial_bh_position1y, y.initial_bh_position1z ] )
        R2 = array( [ y.initial_bh_position2x, y.initial_bh_position2y, y.initial_bh_position2z ] )


    # Estimate the component angular momenta
    try:
        L1 = cross(R1,P1)
        L2 = cross(R2,P2)
    except:
        error('There was an insurmountable problem encountered when trying to load initial binary configuration. For example, %s. The guy at the soup shop says "No soup for you!!"'%red('P1 = '+str(P1)))

    # Extract and store the initial adm energy
    x.madm = y.initial_ADM_energy

    # Store the initial linear momenta
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2) # float( y.initial_separation )
    if abs( y.initial_separation - norm(R1-R2) ) > 1e-1:
        msg = cyan('Warning:')+' The estimated after junk binary separation is significantly different than the value stored in the bbh file: '+yellow('x from calculation = %f, x from bbh file=%f' % (norm(R1-R2),y.initial_separation) )+'. The user should understand whether this is an error or not.'
        x.note += msg
        warning(msg,'bam.py')
    # Let the use know that the binary separation is possibly bad
    if x.b<4:
        msg = cyan('Warning:')+' The estimated initial binary separation is very small. This may be due to an error in the puncture data. You may wish to use the initial binary separation from the bbh file which is %f'%y.initial_separation+'. '
        warning(msg,'bam.py')
        x.note += msg

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    #
    x.valid = True

    # Load irriducible mass data
    irr_mass_file_list = ls(parent(metadata_file_location)+'hmass_2*gz')
    if len(irr_mass_file_list)>0:
        irr_mass_file = irr_mass_file_list[0]
        irr_mass_data,mass_status = smart_load(irr_mass_file)
    else:
        mass_status = False
    # Load spin data
    spin_file_list = ls(parent(metadata_file_location)+'hspin_2*gz')
    if len(spin_file_list)>0:
        spin_file = spin_file_list[0]
        spin_data,spin_status = smart_load(spin_file)
    else:
        spin_status = False
    # Estimate final mass and spin
    if mass_status and spin_status:
        Sf = spin_data[-1,1:]
        irrMf = irr_mass_data[-1,1]
        x.__irrMf__ = irrMf
        irrMf_squared = irrMf**2
        Sf_squared = norm(Sf)**2
        x.mf = sqrt( irrMf_squared + Sf_squared / (4*irrMf_squared) ) / (x.m1+x.m2)
        #
        x.Sf = Sf
        x.Xf = x.Sf/(x.mf*x.mf)
        x.xf = sign(x.Sf[-1])*norm(x.Sf)/(x.mf*x.mf)
    else:
        from numpy import nan
        x.Sf = nan*array([0.0,0.0,0.0])
        x.Xf = nan*array([0.0,0.0,0.0])
        x.mf = nan
        x.xf = nan

    #
    return standard_metadata, raw_metadata

# There are instances when having the extraction radius rather than the extraction paramer is useful.
# Here we define a function which maps between extraction_parameter and extraction radius -- IF such
# a map can be constructed.
def extraction_map( this,                   # this may be an nrsc object or an gwylm object (it must have a raw_metadata attribute )
                    extraction_parameter,   # The extraction parameter that will be converted to radius
                    r_is_tortoise=False,    # Use tortoise coordinate for extraction radius
                    verbose=False):         # verbosity
    '''Given an extraction parameter, return an extraction radius'''

    # NOTE that while some BAM runs have extraction radius information stored in the bbh file in various ways, this does not appear to the case for all simulations. The invariants_modes_r field appears to be more reliable.
    if 'invariants_modes_r' in this.raw_metadata.__dict__:
        _map_ = [ float(k) for k in this.raw_metadata.invariants_modes_r ]
    elif 'extraction_radius' in this.raw_metadata.__dict__:
        # We start from 1 not 0 here becuase the first element should be a string "finite-radius"
        _map_ = [ float(k) for k in this.raw_metadata.extraction_radius[1:] ]

    #
    # print this.raw_metadata.extraction_radius
    # print this.raw_metadata.invariants_modes_r
    # print '>> map = ',_map_
    # raise
    flat_radius = _map_[ extraction_parameter-1 ]

    if r_is_tortoise:
        from numpy import log
        if this.madm:
            adm_mass = this.madm
        else:
            adm_mass = this.raw_metadata.initial_ADM_energy

        alert('Using tortoise coordinate for extraction radius.', verbose)
        try:
            extraction_radius = flat_radius + 2.0 * adm_mass * log( flat_radius / (2.0 * adm_mass) - 1.0 )
        except:
            msg = 'Something has gone awry when computing the tortoise coordinate. Please ensure that the ADM mass is positive definite and that "flat_radius / (2.0 * ADM mass) > 1.0"!'
            error(msg)
    else:
        extraction_radius = flat_radius

    return extraction_radius

# Estimate a good extraction radius and level for an input scentry object from the BAM catalog
def infer_default_level_and_extraction_parameter( this,     # An scentry object
                                                  desired_exraction_radius=None,    # (Optional) The desired extraction radius in M, where M is the initial ADM mass
                                                  verbose=None ):   # Toggel to let the people know
    '''Estimate a good extraction radius and level for an input scentry object from the BAM catalog'''

    # NOTE that input must be scentry object
    # Import useful things
    from glob import glob
    from numpy import array,argmin

    # Handle the extraction radius input
    # NOTE that the default value of 90 is chosen to ensure that there is always a ringdown
    desired_exraction_radius = 90 if desired_exraction_radius is None else desired_exraction_radius

    # Find all l=m=2 waveforms
    search_string = ( this.simdir() + '/Psi4ModeDecomp/*l2.m2*.gz' ).replace('//','/')
    file_list = glob( search_string )

    # For all results
    exr,lev,rad,rad_tort = [],[],[],[]
    for f in file_list:

        # Split filename string to find level and extraction parameter
        f.replace('//','/')
        f = f.split('/')[-1]
        parts = f.split('.') # e.g. "psi3col.r6.l6.l2.m2.gz".split('.')
        exr_,lev_ = int(parts[1][-1]),int(parts[2][-1])
        # Also get related extraction radius (M)
        rad_ = extraction_map( this, exr_, verbose=verbose )
        rad_tort_ = extraction_map (this, exr_, r_is_tortoise=True, verbose=verbose )
        # Append lists
        exr.append(exr_);lev.append(lev_);rad.append(rad_);rad_tort.append(rad_tort_)

    # NOTE that we will use the extraction radius that is closest to desired_exraction_radius (in units of M)

    k = argmin( abs(desired_exraction_radius - array(rad)) )
    extraction_parameter,level = exr[k],lev[k]

    # Also store a dictionary between extraction parameter and extraction radius
    # And a dictionary between the level parameter and extraction radius
    extraction_map_dict = {}
    extraction_map_dict['radius_map'] = { exr[n]:r for n,r in enumerate(rad) }
    extraction_map_dict['radius_map_tortoise'] = { exr[n]:r for n,r in enumerate(rad_tort) }
    extraction_map_dict['level_map'] = { exr[n]:l for n,l in enumerate(lev) }

    # Return answers
    return extraction_parameter,level,extraction_map_dict


#
def learn_source_dynamics(scentry_object,dynamics_times,verbose=False):

    '''
    Based on notebook by Jonathan Thompson, 2019self.

    NOTES
    ---
    * For now, only S, L and J will be calculated and stored. This is to be useful while avoiding faff due to mass ratio convetions.

    USAGE
    ---
    dict_with_source_dynamics = learn_source_dynamics(dynamics_times,verbose=True)

    '''

    # Import usefuls
    from positive import alert,smart_load,lim,spline,find
    from glob import glob as ls
    from nrutils.core.basics import straighten_wfarr
    from numpy import array, cross, linalg
    from os.path import join

    # ---------------------------------- #
    # Load/Calculate Momenta
    # ---------------------------------- #


    # Reference masses
    mass1 = scentry_object.raw_metadata.mass1
    mass2 = scentry_object.raw_metadata.mass2

    # Find puncture data locations
    try:
        puncture_data_1_location = ls( join(scentry_object.simdir(),\
                    'moving_puncture_integrate1*') )[0]
        puncture_data_2_location = ls( join(scentry_object.simdir(), \
                    'moving_puncture_integrate2*') )[0]
    except:
        print(scentry_object.simdir()+\
                    'moving_puncture_integrate1*')
        error('failed to find moving_puncture_integrate* files in ""%s"'%scentry_object.simdir())

    # Location of spin data
    try:
        spin_data_1_location = ls( join(scentry_object.simdir(),\
                'hspin_1*') )[0]
        spin_data_2_location = ls( join(scentry_object.simdir(), \
                    'hspin_2*') )[0]
    except:
        error('failed to find hspin_* files in ""%s"'%scentry_object.simdir())

    # Load puncture and spin data
    puncture_data_1,_ = smart_load( puncture_data_1_location )
    puncture_data_2,_ = smart_load( puncture_data_2_location )
    spin_data_1,_ = smart_load( spin_data_1_location )
    spin_data_2,_ = smart_load( spin_data_2_location )

    # Extract Puncture Locations
    R1_ = array( [  puncture_data_1[:,0],puncture_data_1[:,1],\
            puncture_data_1[:,2],  ] ).T
    R2_ = array( [  puncture_data_2[:,0],puncture_data_2[:,1],\
                puncture_data_2[:,2],  ] ).T

    # Compute component momenta:
    # NOTE that here the shift is actually contained within puncture_data,
    # and NOTE that the shift is -1 times the velocity
    P1 = mass1 * -array( [  puncture_data_1[:,3],puncture_data_1[:,4],\
                puncture_data_1[:,5],  ] ).T
    P2 = mass2 * -array( [  puncture_data_2[:,3],puncture_data_2[:,4],\
                puncture_data_2[:,5],  ] ).T

    # Reference spins
    S1_ = array( [  spin_data_1[:,1],spin_data_1[:,2],\
                spin_data_1[:,3],  ] )
    S2_ = array( [  spin_data_2[:,1],spin_data_2[:,2],\
                spin_data_2[:,3],  ] )

    # Estimate the component angular momenta
    L1_ = cross(R1_,P1)
    L2_ = cross(R2_,P2)
    L_ = L1_+L2_

    # Time values
    L_times  = puncture_data_1[:, -1]
    S1_times = spin_data_1[:, 0]
    S2_times = spin_data_2[:, 0]

    #
    def straighten(times,vec):
        vx,vy,vz = vec if vec.shape[0]==3 else vec.T
        varr = array( [ times,vx,vy,vz ] ).T
        varr = straighten_wfarr( varr, verbose=True )
        new_times,vx,vy,vz = varr.T
        new_vec = array( [vx,vy,vz] ).T
        return new_times, new_vec.T

    #
    S1_times,S1_ = straighten(S1_times,S1_)
    S2_times,S2_ = straighten(S2_times,S2_)
    #
    L1_times,L1_ = straighten(L_times,L1_)
    L2_times,L2_ = straighten(L_times,L2_)
    #
    R1_times,R1_ = straighten(L_times,R1_)
    R2_times,R2_ = straighten(L_times,R2_)

    #
    abs_dr = linalg.norm( (R2_-R1_).T, axis=1 )
    r0 = 3.1
    # from matplotlib.pyplot import plot,show,axvline,axhline
    # plot( R1_times, abs_dr )
    internal_t_max = R1_times[ find(abs_dr < r0)[0] ]
    # axhline( r0, color='k' )
    # axvline( internal_t_max, color='k' )
    # print internal_t_max, max(dynamics_times)
    # show()
    if internal_t_max<max(dynamics_times):
        dynamics_times = dynamics_times[ dynamics_times<internal_t_max ]
    def mask( t ):
        msk = (t>=min(dynamics_times)) & (t<=max(dynamics_times) )
        return msk

    # Interpolate everything of use. Some care is taken with the spins as the data files may have sligtly different time series.
    dynamics_times = dynamics_times[ dynamics_times < max(lim(L_times)[-1],lim(S1_times)[-1]) ]
    #
    msk = mask( S1_times )
    S1 = mass1**2 * array(  [ spline(S1_times[msk],s[msk])(dynamics_times) for s in S1_ ]  ).T
    msk = mask( S2_times )
    S2 = mass2**2 * array(  [ spline(S2_times[msk],s[msk])(dynamics_times) for s in S2_ ]  ).T
    #
    msk = mask( L1_times )
    L1 = array(  [ spline(L1_times[msk],l[msk])(dynamics_times) for l in L1_ ]  ).T
    msk = mask( L2_times )
    L2 = array(  [ spline(L2_times[msk],l[msk])(dynamics_times) for l in L2_ ]  ).T
    #
    msk = mask( R1_times )
    R1 = array(  [ spline(R1_times[msk],l[msk])(dynamics_times) for l in R1_ ]  ).T
    msk = mask( R2_times )
    R2 = array(  [ spline(R2_times[msk],l[msk])(dynamics_times) for l in R2_ ]  ).T

    # Total angular momenta
    S = S1+S2   # Spin
    L = L1+L2   # Spin
    J = L+S     # Orbital  Spin

    # Save everything in a standard dictionary
    # HERE we swap 1,2 labels to be consistent with nrutls' internal convention
    foo = {}
    # ORBITAL MOMENTA
    foo['L2'] = L1
    foo['L1'] = L2
    foo['L'] = L
    # SPIN MOMENTA
    foo['S2'] = S1
    foo['S1'] = S2
    foo['S'] = S
    # TOTAL MOMENTA
    foo['J'] = J
    # TRAJECTORIES
    foo['R2'] = R1
    foo['R1'] = R2
    # DYNAMICS TIMES USED
    foo['dynamics_times'] = dynamics_times

    # Let's go! :D
    ans = foo
    return ans
