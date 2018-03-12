# Import usefuls
from positive import *
from nrutils.core.basics import *

# Make the workflow a class
class make_pnnr_hybrid:

    # The class constructor
    def __init__( this,             # The current object
                  gwylmo,           # An instance of gwylm class containing initial system params and NR data (psi4)
                  pn_w_orb_min = None, # Min Orbital freq for PN generation
                  pn_w_orb_max = None, # Max Orbital freq for PN generation
                  verbose = False,  # Toggle for letting the people know
                  plot = False,     # Toggle for plotting
                  kind = None,      # psi4,strain -- kind of data that will be used for hybridization
                  **kwargs          # Args for PN generation
                ):

        #
        cname = 'make_pnnr_hybrid'

        # Validate inputs
        if verbose: alert('Validating inputs',cname,header=True)
        this.__validate_inputs__(gwylmo,pn_w_orb_min,pn_w_orb_max,kind,verbose)

        # Access or generate PN waveform
        if verbose: alert('Generating PN multipoles',cname,header=True)
        this.__generate_pn__()

        # Determine hybridization parameters for the l=m=2 multipole
        if verbose: alert('Calculating l=m=2 hybrid parameters',cname,header=True)
        # this.__calc_l2m2_hybrid_params__()

        # Apply optimal hybrid parameters to all multipoles
        # this.__calc_multipole_hybrids__()


    # Determine hybridization parameters for the l=m=2 multipole
    def __calc_l2m2_hybrid_params__(this,plot=False):

        # Import usefuls
        from scipy.optimize import minimize
        from numpy import pi,linspace,mean,std,unwrap,angle,exp

        # Setup plotting
        __plot__ = plot
        if __plot__:
            # Import usefuls
            from matplotlib.pyplot import plot,ylim,xlim,xlabel,ylabel,\
            figure,figaspect,yscale,xscale,axvline,axhline,show,title
            alert('Plots will be generated.',verbose=this.verbose)

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Extract domain and range values of interest
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        kind = this.__kind__
        pno = this.__pn__
        nro = this.gwylmo
        pn_t,pn_y,nr_t,nr_y = pno.pn_gwylmo.t,pno.pn_gwylmo[2,2][kind].y,nro.t,nro[2,2][kind].y
        alert('The l=m=2 %s will be used for determining (initial) optimal hybrid params.'%cyan(kind),verbose=this.verbose)

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Format data in a common way
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        alert('Aligning formats of NR and PN data',verbose=this.verbose)
        t,pn_y,nr_y = format_align(pn_t,pn_y,nr_t,nr_y,center_domains=True)

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Align data for initial guesses
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        t,pn_,t,nr_,foo = corr_align(t,pn_y,t,nr_y)
        alert('Storing hybrid time series.',verbose=this.verbose)
        this.t = t

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Calculate time resolution
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        dt = t[1]-t[0]
        this.dt = dt

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Define limits of fitting region
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        N = 2 # Width of hybrid region in number of cycles (approx)
        alert('We will use %i cycles for the hybridization region\'s width.'%N,verbose=this.verbose)
        T1 = (nro.t-nro.t[0])[ nro.startindex ]
        T2 = (nro.t-nro.t[0])[ nro.startindex + int(N*(2*pi)/(nro.wstart_pn/2)) ]
        mask = (t>=T1) & (t<=T2)

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Define methods for additional alignment
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Method: Time shift the PN data
        pn_time_shift = lambda TSHIFT,PN=pn_y,METH=None: tshift( t, PN, TSHIFT,method=METH ) # roll( pn_y, int(t0/dt) )
        # Method: Align in phase using average phase difference
        def pn_phi_align(PN,NR,return_phi=False,MSK=mask,phi0=None):
            # Apply an optimal phase shift
            getphi = lambda X: unwrap(angle(X))
            if phi0 == None: phi0 = mean( (getphi(PN)-getphi(NR))[MSK] )
            if return_phi:
                return PN*exp(-1j*phi0), phi0
            else:
                return PN*exp(-1j*phi0)
        # Method: Calculate estimator
        estimartor_fun = lambda PN,NR=nr_y,MSK=mask: abs( std(PN[MSK]-NR[MSK])/std(NR[MSK]) )
        # Store usefuls
        alert('Storing shifting functions for future reference.',verbose=this.verbose)
        this.__pn_time_shift__ = pn_time_shift
        this.__estimator_fun__ = estimartor_fun
        this.__pn_phi_align__ = pn_phi_align

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Define work function for optimizer
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        def work( t0 ):
            # Shift the PN waveform by t0
            pn_y_ = pn_time_shift(t0)
            # Apply an optimal phase shift
            pn_y_ = pn_phi_align(pn_y_,nr_y)
            # Compute a residual error statistic
            frmse = estimartor_fun( pn_y_ )
            # Return estimator
            return frmse

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Numerically Determine Optimal Tims Shift
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #

        alert('Finding optimal time-shift using scipy.optimize.minimize',verbose=this.verbose)
        t0_guess = len(t)-foo['domain_shift']
        bear = minimize( work, t0_guess )
        est0 = bear.fun
        t0 = bear.x[0]

        if __plot__:
            t0_space = linspace(t0-200*dt,t0+200*dt,21)
            figure()
            plot( t0_space, map(work,t0_space) )
            xlim(lim(t0_space))
            axvline(t0,color='r',ls='--')
            axvline(t0_guess,color='k',ls='-',alpha=0.3)
            title('(%f,%f)'%(t0,t0_guess))

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Apply optimal parameters
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #

        # Define general functoin for applying hybrid params
        def apply_hyb_params_to_pn(PN,NR,T0,MSK=mask,TSMETH=None,phi0=None):
            # Apply optimal time shift
            PN_ = this.__pn_time_shift__( T0,PN=PN,METH=TSMETH )
            # Compute and apply an optimal phase shift
            PN_,phi0 = pn_phi_align(PN_,NR,MSK=mask,phi0=phi0,return_phi=True)
            # Return deliverables
            return PN_,phi0
        # Store method
        this.__apply_hyb_params_to_pn__ = apply_hyb_params_to_pn

        # Apply optimal time shift
        pn_y_,phi0_22 = this.__apply_hyb_params_to_pn__(pn_y,nr_y,t0,MSK=mask)

        if __plot__:
            figure( figsize=1*figaspect(1.0/7) )
            plot( t, pn_y_.real )
            plot( t, nr_y.real )
            plot( t, abs(nr_y) )
            plot( t, abs(pn_y_) )
            xlim( [100,2500] )
            axvline(T1,color='b')
            axvline(T2,color='b')
            yscale('log',nonposy='clip')
            ylim([1e-6,5e-3])

        # Store optimals
        alert('Storing optimal params to this.optimal_hybrid_params',verbose=this.verbose)
        this.optimal_hybrid_params = { 't0':t0, 'phi0_22':phi0_22, 'mask':mask, 'T1':T1, 'T2':T2 }
        if this.verbose: print this.optimal_hybrid_params


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
                  wM_min=this.pn_w_orb_min,
                  wM_max=this.pn_w_orb_max,
                  sceo = y.__scentry__ )

        # Store to current object
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
    def __validate_inputs__(this,gwylmo,pn_w_orb_min,pn_w_orb_max,kind,verbose):

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

        # Handle kind
        if kind is None: kind = 'psi4'
        this.__kind__ = kind

        # Store ,min and max orbital frequency for PN generation
        if pn_w_orb_min is None:
            strain_w_orb_min = gwylmo.lm[2,2]['strain'].dphi[gwylmo.remnant['mask']][0]/2
            pn_w_orb_min = 0.65 * strain_w_orb_min
            if this.verbose: alert('Using default values for PN w_orb starting frequency based on strain waveform: %s'%(yellow('%f'%pn_w_orb_min)))
        if pn_w_orb_max is None:
            strain_w_orb_max = gwylmo.lm[2,2]['strain'].dphi[gwylmo.remnant['mask']][0]/2
            pn_w_orb_max = 4.0 * strain_w_orb_max
            if this.verbose: alert('Using default values for PN w_orb end frequency based on strain waveform: %s'%(yellow('%f'%pn_w_orb_max)))

        # Store PN start and end frequency
        this.pn_w_orb_min = pn_w_orb_min
        this.pn_w_orb_max = pn_w_orb_max
        if this.verbose:
            alert('PN w_orb MIN frequency is %s (i.e. w_orb*M_init)'%(yellow('%f'%pn_w_orb_min)))
            alert('PN w_orb MAX frequency is %s (i.e. w_orb*M_init)'%(yellow('%f'%pn_w_orb_max)))

        # Store select inputs
        this.gwylmo = gwylmo
