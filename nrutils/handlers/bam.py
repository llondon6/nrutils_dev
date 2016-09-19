
#
from nrutils.core.basics import smart_object,parent,blue,smart_load,green,alert,error,warning,yellow,green,bold,cyan
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt
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
    status = len( ls( run_dir + '/Psi4ModeDecomp/psi3col*l2.m2.gz' ) ) > 0

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
def learn_metadata( metadata_file_location ):

    # Try to load the related par file
    par_file_location = metadata_file_location[:-3]+'par'

    #
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
    x.isafterjunk = not ( (not S1) or ( not S2 ) )
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
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    # Extract and store the initial adm energy
    x.madm = y.initial_ADM_energy

    # Store the initial linear momenta
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2) # float( y.initial_separation )
    if abs( y.initial_separation - norm(R1-R2) ) > 1e-1:
        msg = cyan('Warning:')+' The estimated after junk binary separation is significantly different than the value stored in the bbh file: '+yellow('x from calculation = %f, x from bbh file=%f' % (norm(R1-R2),y.initial_separation) )+'. The user should understand whether this is an erorr or not.'
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
        x.mf = sqrt( irrMf**2 + norm(Sf/irrMf)**2 )
        #
        x.Sf = Sf
        x.xf = norm(x.Sf)/(x.mf*x.mf)
    else:
        x.Sf = array([0.0,0.0,0.0])
        x.mf = 0.0
        x.xf = array([0.0,0.0,0.0])


    # True if ectraction parameter is extraction radius
    x.extraction_parameter_is_radius = False

    #
    return standard_metadata, raw_metadata

# # Create a file-name string based upon l,m and the extraction parameter(s)
# # NOTE that the method name and inputs must conform to uniform name and input ordering
# def make_datafilename( gwylm_object,                    # gwylm object
#                        l,                               # l spherical index
#                        m,                               # m index; |m|<=l
#                        extraction_parameter = None ):   # dictionary of extraction information
#
#     # Validate the extraction parameter for BAM simulations
#     if isinstance( extraction_parameter, list ):
#
#     # Validate l and m inputs
#     if not isinstance(l, (int,float) ):
#         raise ValueError('l input must be int or float, but %s found' % (type(l).__name__) )
#     if not isinstance(m, (int,float) ):
#         raise ValueError('m in input must be int or float, but %s found' % (type(m).__name__) )
#
#     # Create the filename string using inputs
#
#
#     # Output the filename string
