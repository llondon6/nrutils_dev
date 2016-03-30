
#
from nrutils.core.basics import smart_object,parent,blue,smart_load
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

#
def learn_metadata( metadata_file_location ):

    #
    raw_metadata = smart_object( metadata_file_location )
    # shortand
    y = raw_metadata

    # # Useful for debuggin -- show what's in y
    # y.show()

    #
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
    x.m1 = y.mass1
    x.m2 = y.mass2

    #
    P1 = array( [ y.initial_bh_momentum1x, y.initial_bh_momentum1y, y.initial_bh_momentum1z ] )
    P2 = array( [ y.initial_bh_momentum2x, y.initial_bh_momentum2y, y.initial_bh_momentum2z ] )

    #
    S1 = array( [ y.initial_bh_spin1x, y.initial_bh_spin1y, y.initial_bh_spin1z ] )
    S2 = array( [ y.initial_bh_spin2x, y.initial_bh_spin2y, y.initial_bh_spin2z ] )

    #
    R1 = array( [ y.initial_bh_position1x, y.initial_bh_position1y, y.initial_bh_position1z ] )
    R2 = array( [ y.initial_bh_position2x, y.initial_bh_position2y, y.initial_bh_position2z ] )

    #
    x.note = ''

    #
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    #
    x.madm = y.initial_ADM_energy

    #
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    #
    x.b = float( y.initial_separation )
    if abs( x.b - norm(R1-R2) ) > 1e-4:
        msg = '(!!) Inconsistent assignment of initial separation: \n\t\tx = %f\n\t\tdR=%f' % (x.b,norm(R1-R2))
        raise ValueError(msg)

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
