# Import usefuls
from positive import *
from nrutils.core.basics import *

# Make the workflow a class
class make_pnnr_hybrid:

    # The class constructor
    def __init__( this,             # The current object
                  gwylmo,           # An instance of gwylm class containing initial system params and NR data (psi4)
                  verbose = False,  # Toggle for letting the people know
                  **kwargs          # Args for PN generation
                ):

        #
        cname = 'make_pnnr_hybrid'

        # Validate inputs
        if verbose: alert('Validating inputs',cname,header=True)
        this.__validate_inputs__(gwylmo,verbose)

        # Access or generate PN waveform
        if verbose: alert('Generating PN multipoles',cname,header=True)
        this.__generate_pn__()

        # Determine hybridization parameters for the l=m=2 multipole
        if verbose: alert('Generating PN multipoles',cname,header=True)
        this.__calc_l2m2_hybrid_params__()

        # Apply optimal hybrid parameters to all multipoles
        # this.__calc_multipole_hybrids__()


    # Determine hybridization parameters for the l=m=2 multipole
    def __calc_l2m2_hybrid_params__(this):

        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#
        # Estimate optimal params -- Initial guesses
        #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~#

        # Estimate time shift (T) from frequency content
        T = 0
        # Given T, optimize orbital phase (handle polarization shift simulteneously)
        phi,psi = 0,0
        # Do not optimize amplitude

        # Store initial guess
        this.__guess_hybrid_params__ = {'T':T,'phi':phi,'psi':psi}

        # Given the initial guess , find the optimal values
        this.optimal_hybrid_params = 0# a dictionary


    # Create an instance of the PN class
    def __generate_pn__(this):

        # Define initial binary parameters
        y = this.gwylmo
        m1 = y.m1
        m2 = y.m2
        X1 = y.X1
        X2 = y.X2

        # Initiate PN object
        pno = pn( m1,m2,X1,X2,
                  wM_max=1.05*y.lm[2,2]['strain'].dphi[y.remnant['mask']][0],
                  wM_min=0.7*y.lm[2,2]['strain'].dphi[y.remnant['mask']][0]/2,
                  sceo = y.__scentry__ )

        # Store to current obejct
        this.__pn__ = pno

        #
        return None


    # Apply optimal hybrid parameters to all multipoles
    def __calc_multipole_hybrids__(this):

        #
        psi = this.gwylmo.psi
        _psi = []
        for y in psi:
            #
            _y = this.__calc_pnnr_gwf__( y, this.optimal_hybrid_params )
            # Store the new
            _psi.append( _y )

        #


        #
        return None


    # Validative constructor for class workflow
    def __validate_inputs__(this,gwylmo,verbose):

        # Let the people know
        this.verbose = verbose
        if verbose:
            alert('Verbose mode ON.')

        # Validate gwylmo type
        if not isgwylm( gwylmo ):
            error('First input is not a member of the gwylmo class. Please check the input type.')
        else:
            if this.verbose: alert('Valid gwylm object found. Its simulation name is "%s".'%yellow(gwylmo.simname))

        # Calculate radiated quantities for the input gwylmo
        gwylmo.__calc_radiated_quantities__()

        # Store select inputs
        this.gwylmo = gwylmo
