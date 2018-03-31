# Import usefuls
from positive import *
from nrutils.core.nrsc import gwylm
from nrutils.core.basics import *

# Make the workflow a class
class make_pnnr_hybrid(gwylm):

    # The class constructor
    def __init__( this,                 # The current object
                  gwylmo,               # An instance of gwylm class containing initial system params and NR data (psi4)
                  pn_w_orb_min = None,  # Min Orbital freq for PN generation
                  pn_w_orb_max = None,  # Max Orbital freq for PN generation
                  verbose = False,      # Toggle for letting the people know
                  plot = False,         # Toggle for plotting
                  kind = None,          # psi4,strain -- kind of data that will be used for hybridization
                  aggressive = True,    # Toggle for agressiveness of hybridization workflow; see doc
                  **kwargs              # Args for PN generation
                ):

        '''
        NOTE that this class multiple tracks based on the aggressive attribute:

        this.aggressive = False             Self consistency of PN and NR data are strictly enforced.
        this.aggressive = True (Default)    Self consistency of PN amplitudes is enforced, but phases are aligned to NR.
        this.aggressive = 2                 PN amplitudes and phases are aligned to NR, and may not be exactly consistent
                                            with PN theory in the low frequency limit.
        '''

        #
        cname = 'make_pnnr_hybrid'

        # Validate inputs
        if verbose: alert('Validating inputs',cname,header=True)
        this.__validate_inputs__(gwylmo,pn_w_orb_min,pn_w_orb_max,kind,aggressive,verbose)

        # Access or generate PN waveform
        if verbose: alert('Generating PN multipoles',cname,header=True)
        this.__generate_pn__()

        # Determine hybridization parameters for the l=m=2 multipole
        if verbose: alert('Calculating l=m=2 hybrid parameters',cname,header=True)
        this.__calc_l2m2_hybrid_params__()

        # Apply optimal hybrid parameters to all multipoles
        if verbose: alert('Calculating multipole %s hybrids'%(cyan(this.__kind__)),cname,header=True)
        this.__calc_multipole_hybrids__(plot=plot)


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
        alert('Storing hybrid time series.',verbose=this.verbose)
        this.t = t

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Align data for initial guesses
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        alert('Using cross-correlation to estimate optimal params.',verbose=this.verbose)
        t,pn_,t,nr_,foo = corr_align(t,abs(pn_y),t,abs(nr_y))

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Calculate time resolution
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        dt = t[1]-t[0]
        this.dt = dt

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        # Define limits of fitting region
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
        N = 3 # Width of hybrid region in number of cycles (approx)
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
        k0 = round(t0/dt)

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
        this.optimal_hybrid_params = { 't0':t0, 'dt':dt, 'k0':k0, 'phi0_22':phi0_22, 'mask':mask, 'T1':T1, 'T2':T2, 'hybrid_cycles':N }
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
        alert('Determining multipole to consider for hybridization by taking the set intersection of those present in NR and PN data.',verbose=this.verbose)
        this.lmlist = sorted( list(set( this.gwylmo.lm.keys() ).intersection( pno.pn_gwylmo.lm.keys() )) )
        if this.verbose:
            alert('Hybrid waveforms will be constructed for the following multipoles:')
            print this.lmlist

        #
        return None


    # Apply optimal hybrid parameters to all multipoles
    def __calc_multipole_hybrids__(this,plot=None,lmlist=None):

        # Import usefuls
        from numpy import roll,exp,array,pi,mod

        # Handle input toggles
        if lmlist is None: lmlist = this.lmlist

        # Copy input gwylmo, and reset multipole holders
        this.hybrid_gwylmo = this.gwylmo.copy()
        this.hybrid_gwylmo.ylm = []
        this.hybrid_gwylmo.flm = []
        this.hybrid_gwylmo.hlm = []

        # Define params for taper
        # NOTE that here we slowly turn on (or taper) the start of the PN waveform to assist with future fourier transform of the data
        dt = this.t[1]-this.t[0]
        k0 = mod(int(this.optimal_hybrid_params['k0']),len(this.t))
        k1 = k0 + 4 * int( (2*pi/this.__pn__.wM_min)/(dt) )
        state = [ k0, k1 ]
        window = maketaper(this.t,state)
        window[:k0] = 1

        #
        for kind in [this.__kind__]:
            for lm in lmlist:
                # Let the people know
                alert('Creating hybrid for: %s'%(green(str(lm))),verbose=this.verbose,heading=True,pattern='++')
                # Get hybrid data
                t,amp,phi = this.__calc_single_multipole_hybrid__(lm,kind,plot=plot)
                # Apply a taper to the amplitude to assist fourier transforming in gwf creation
                amp *= window
                # Shift waveform features to put start of PN at beginning of array
                amp = roll( amp, -k0 )
                phi = roll( phi, -k0 )
                # Create waveform array for gwf creation
                _y = amp*exp(1j*phi)
                wfarr = array( [t, _y.real, _y.imag] ).T
                # Create gwf object and store to current object
                gwfo = this.gwylmo[lm][kind].copy()
                gwfo.setfields( wfarr )
                if kind == 'psi4':
                    this.hybrid_gwylmo.ylm.append( gwfo )
                elif kind == 'strain':
                    this.hybrid_gwylmo.hlm.append( gwfo )
                else:
                    error('only psi4 and strain handled at the moment')
                if plot:
                    from matplotlib.pyplot import show
                    # tl = array([this.optimal_hybrid_params['T1'],this.optimal_hybrid_params['T2']])+dt*k0
                    # tl = t[-1]-tl
                    # tl.sort()
                    # print tl
                    gwfo.plot(domain='time',sizescale=2)
                    gwfo.plot(domain='freq',sizescale=2,ref_gwf=this.gwylmo[lm][kind],labels=('PN-NR Hybrid','NR'))
                    show()

        #
        this.hybrid_gwylmo.__lmlist__ = this.lmlist
        this.hybrid_gwylmo.__input_lmlist__ = this.lmlist
        this.hybrid_gwylmo.__curate__()
        this.hybrid_gwylmo.wstart = this.hybrid_gwylmo.wstart_pn = this.__pn__.wM_min*2

        #
        return None


    # Generate a model trained to pn at low freqs and nr at high freqs
    def __calc_single_multipole_hybrid__(this,lm,kind,plot=False):

        # Import usefuls
        from numpy import zeros_like,argmax,arange,diff,roll,unwrap,mod,exp

        # Create shorthand for useful information
        l,m = lm
        T1 = this.optimal_hybrid_params['T1']
        T2 = this.optimal_hybrid_params['T2']

        # Get the time and phase aligned waveforms
        foo = this.__get_aligned_nr_pn_amp_phase__( lm,kind,plot=plot )
        nr_amp = foo['nr_amp']
        nr_phi = foo['nr_phi']
        pn_amp = foo['pn_amp']
        pn_phi = foo['pn_phi']
        t = foo['t']
        pn_t_amp_max = t[ argmax(pn_amp) ]

        #
        stride = 4*(T2-T1)
        pnmask = (t>=max(T1-stride,0)) & (t<=T2)
        nrmask = (t>=t[foo['nr_smoothest_mask']][0]) & (t<=min(pn_t_amp_max,t[foo['nr_smoothest_mask']][0]+stride))

        # Deermine if a bridge model is needed
        # NOTE that a bridge model (between PN and NR regions over the noisey NR) is only needed if the smooth NR data does not contain the region bounded by [T1,T2].
        BRIDGE = t[nrmask][0] > T1

        #
        if BRIDGE:

            # Create training data for bridge models
            __bridge_t = t.copy()
            __bridge_t   = __bridge_t[ pnmask | nrmask ]
            #
            __bridge_amp = zeros_like(t)
            __bridge_amp[ pnmask ] = pn_amp[ pnmask ]
            __bridge_amp[ nrmask ] = nr_amp[ nrmask ]
            __bridge_amp = __bridge_amp[ pnmask | nrmask ]
            #
            __bridge_phi = zeros_like(t)
            __bridge_phi[ pnmask ] = pn_phi[ pnmask ]
            __bridge_phi[ nrmask ] = nr_phi[ nrmask ]
            __bridge_phi = __bridge_phi[ pnmask | nrmask ]

            # Model the AMPLITUDE with fixed symbols; TODO -- check for poles which seem to appear when both NR and PN are poor
            rope = gmvrfit( __bridge_t, __bridge_amp )
            # rope = mvrfit( __bridge_t, __bridge_amp, numerator_symbols=[('00')], denominator_symbols=['00','000'] )

            # Model the PHASE with fixed symbols
            plank = mvpolyfit( __bridge_t, __bridge_phi, basis_symbols=['K','0','000'] )
            # plank = gmvpfit( __bridge_t, __bridge_phi ) # NOTE -- this works but is slower

            # Create blending window towards connecting models with NR
            bridge_amp = rope.eval(t)
            bridge_phi = plank.eval(t)
            bridge_state = lim( arange(len(t))[ nrmask ] )
            window = maketaper( t, bridge_state )

            # Extend NR towards PN using models
            u = window
            extended_nr_amp = (1-u)*bridge_amp + u*nr_amp
            extended_nr_phi = (1-u)*bridge_phi + u*nr_phi

        else:

            #
            warning('The NR data appares to be sufficiently smooth. No bridge model will be used.',verbose=this.verbose)

            # No extension to the left (i.e. bridge) is required
            extended_nr_amp = nr_amp
            extended_nr_phi = nr_phi

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # Create final hybrid multipole by repeating the process above for the extended nr and pn
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # Select hybridization region & make related taper
        mask = (t>=T1) & (t<=T2)
        hybrid_state = lim( arange(len(t))[ mask ] )
        u = maketaper( t, hybrid_state )
        # Currently, the taper removes PN data that may be to the right of NR data due to wrapping around when time shifting, so we correct for this by turning the taper off
        k_pn_start = int(  mod(this.optimal_hybrid_params['k0'],len(t))  )
        u[k_pn_start:] = 0
        hybrid_amp = (1-u)*pn_amp + u*extended_nr_amp
        hybrid_phi = (1-u)*pn_phi + u*extended_nr_phi
        # hybrid_amp = roll(hybrid_amp,k_pn_start)
        # hybrid_phi = unwrap(roll(hybrid_phi,k_pn_start))

        # Plotting
        fig,(ax1,ax2) = None,(None,None)
        if plot:
            # Import usefuls
            from matplotlib.pyplot import figure,figaspect,plot,xlim,ylim,xscale,subplot,show
            from matplotlib.pyplot import yscale,axvline,axvspan,xlabel,ylabel,title,yscale,legend
            # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
            # Informative plots of intermediate information
            # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
            # Create figure
            fig = figure( figsize=3*figaspect(4.0/7) )
            #
            ax1 = subplot(2,1,1)
            # Show training regions
            if BRIDGE:
                axvspan( min(t[pnmask]),max(t[pnmask]), color='cyan', alpha = 0.15, label='Bridge Training Region' )
                axvspan( min(t[nrmask]),max(t[nrmask]), color='cyan', alpha = 0.15 )
                axvline(min(t[pnmask]),color='c',ls='--')
                axvline(max(t[pnmask]),color='c',ls='--')
                axvline(min(t[nrmask]),color='c',ls='--')
                axvline(max(t[nrmask]),color='c',ls='--')
            # Plot amplitudes
            plot( t, nr_amp, label='NR', alpha=0.2, lw=4, ls='--' )
            if BRIDGE: plot( t,rope.eval(t), lw=2,label='Bridge Model' )
            plot( t, hybrid_amp, '-', label='Hybrid' )
            plot( t, pn_amp, label='PN', linewidth=2, ls='--', color='k', alpha=0.5 )
            # Set plot limits
            yscale('log')
            ylim( min(pn_amp[pn_amp>0]),max(lim( nr_amp[foo['nr_smoothest_mask']], dilate=0.1)) )
            xlim( 0,max(t[foo['nr_smoothest_mask']]) )
            # Set axis labels and legend
            ylabel(r'$|\psi_{%i%i}|$'%(l,m))
            legend(frameon=True,framealpha=1,edgecolor='k',fancybox=False)
            #
            ax2 = subplot(2,1,2)
            # Show training regions
            if BRIDGE:
                axvspan( min(t[pnmask]),max(t[pnmask]), color='cyan', alpha = 0.15, label='Bridge Training Region' )
                axvspan( min(t[nrmask]),max(t[nrmask]), color='cyan', alpha = 0.15 )
                axvline(min(t[pnmask]),color='c',ls='--')
                axvline(max(t[pnmask]),color='c',ls='--')
                axvline(min(t[nrmask]),color='c',ls='--')
                axvline(max(t[nrmask]),color='c',ls='--')
            # Plot phases
            plot( t, nr_phi, label='NR', alpha=0.2, lw=4, ls='--' )
            if BRIDGE: plot( t,plank.eval(t), lw=2,label='Bridge Model' )
            plot( t, hybrid_phi, '-' )
            plot( t, pn_phi, label='PN', linewidth=2, ls='--', color='k', alpha=0.5 )
            # Set plot limits
            ylim( min(pn_phi),max(lim( nr_phi[foo['nr_smoothest_mask']], dilate=0.1)) )
            xlim( 0,max(t[foo['nr_smoothest_mask']]) )
            # plot( t, window*diff(ylim())+min(ylim()), color='k',alpha=0.3 )
            # plot( t, (1-window)*diff(ylim())+min(ylim()), color='k',alpha=0.3 )
            # Set axis labels
            xlabel(r'$t/M$'); ylabel(r'$\phi_{%i%i}=\arg(\psi_{%i%i})$'%(l,m,l,m))
            # # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
            # # Informative plots of final hybrid and original pn+nr
            # # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
            # # Create figure
            # fig = figure( figsize=1.5*figaspect(4.0/7) )
            # #
            # ax1 = subplot(2,1,1)
            # # plot( t, nr_amp, label='NR', alpha = 0.5 )
            # plot( t, roll(hybrid_amp,int(this.optimal_hybrid_params['k0'])), label='Hybrid' )
            # plot( t, roll(pn_amp,int(this.optimal_hybrid_params['k0'])), label='PN' )
            # # Set plot limits
            # yscale('log')
            # # ylim( min(pn_amp[pn_amp>0]),max(lim( nr_amp[foo['nr_smoothest_mask']], dilate=0.1)) )
            # # xlim( 0,max(t[foo['nr_smoothest_mask']]) )
            # # Set axis labels and legend
            # ylabel(r'$|\psi_{%i%i}|$'%(l,m))
            # legend()
            # #
            # ax2 = subplot(2,1,2)
            # # plot( t, nr_phi, label='NR', alpha = 0.5 )
            # plot( t, unwrap(roll(hybrid_phi,int(this.optimal_hybrid_params['k0']))), label='Hybrid' )
            # plot( t, unwrap(roll(pn_phi,int(this.optimal_hybrid_params['k0']))), label='PN' )
            # # Set plot limits
            # # ylim( min(pn_phi),max(lim( nr_phi[foo['nr_smoothest_mask']], dilate=0.1)) )
            # # xlim( 0,max(t[foo['nr_smoothest_mask']]) )
            # # Set axis labels
            # xlabel(r'$t/M$'); ylabel(r'$\phi_{%i%i}=\arg(\psi_{%i%i})$'%(l,m,l,m))

        # Output data for further processing
        return t,hybrid_amp,hybrid_phi


    # Given optimal hybrid params for l=m=2, begin to estimate optimal phase alignment for PN
    def __get_aligned_nr_pn_amp_phase__(this,lm,kind,plot=False,verbose=False):
        '''
        Given optimal hybrid params for l=m=2, begin to estimate optimal phase alignment for PN
        '''
        # Import usefuls
        from numpy import argmax,array,unwrap,angle,pi,mean,exp,arange
        # Unpack l and m
        l,m = lm
        # Create shorthand for useful information
        k0 = this.optimal_hybrid_params['k0']
        dt = this.optimal_hybrid_params['dt']
        mask = this.optimal_hybrid_params['mask']
        T1 = this.optimal_hybrid_params['T1']
        T2 = this.optimal_hybrid_params['T2']
        phi22 = this.optimal_hybrid_params['phi0_22']
        # Find the peak location for l=m=2 to be used for masking below
        k_amp_max_22 = argmax( this.__unpacklm__((2,2),kind)[1] )
        # Get the format aligned data for this multipole
        t,nr_y,pn_y = this.__unpacklm__(lm,kind)
        # Apply optimal time shift, NOTE the rescaling of phi22
        pn_y,phi0 = this.__apply_hyb_params_to_pn__(pn_y,nr_y,k0*dt,MSK=mask,TSMETH='index',phi0=m*phi22/2)
        # Begin phase alignment
        __getphi__= lambda X: unwrap(angle(X))
        # Get phases aligned in mask region
        def get_aligned_phi(a,b,__mask__):
            '''Get phases aligned in mask region'''
            A = __getphi__(a)
            B = __getphi__(b)
            L = pi/2
            n = round(  mean(B[__mask__]-A[__mask__]) / L  )
            B -= n*L
            return A,B
        # Get NR phase
        nr_phi = __getphi__( nr_y )
        # Find a mask for the smoothest part
        nr_smoothest_mask_amp = smoothest_part( abs(nr_y)[:k_amp_max_22] )
        nr_smoothest_mask_pha = smoothest_part( nr_phi[:k_amp_max_22] )
        # nr_smoothest_mask_pha = nr_smoothest_mask_amp
        # Take the union of the amplitude and phase masks
        nr_smoothest_mask = arange( min(min(nr_smoothest_mask_amp),min(nr_smoothest_mask_pha)), min(max(nr_smoothest_mask_amp),max(nr_smoothest_mask_pha)) )
        # Get the aligned phases (NOTE that nr_phi should not have changed)
        # NOTE that below we wish to use T1 as the start of the alignment region if the entire NR waveform is erroneously deemed smooth (e.g. due to settings in smoothest_part)
        nr_phi,pn_phi = get_aligned_phi( nr_y, pn_y, nr_smoothest_mask[:10] if t[nr_smoothest_mask][0]>T1 else nr_smoothest_mask[ t[nr_smoothest_mask]>=T1 ][:10] )
        # Get Amplitudes
        nr_amp,pn_amp = abs(nr_y),abs(pn_y)
        # --
        # Impose scaling of PN amplitude to match NR to accomodate limitations of PN
        # NOTE that this is an agressive approach
        if this.aggressive:
            # >>>
            T1k = max( t[nr_smoothest_mask[0]], T1 )
            T2k = T1k + (T2-T1)/this.optimal_hybrid_params['hybrid_cycles']
            k = (t>=T1k) & (t<=T2k)
            scale_factor = mean( nr_amp[k]/pn_amp[k] )
            if this.aggressive==2:
                warning('A scale factor of %f is applied to the PN amplitude.'%scale_factor,verbose=this.verbose)
                pn_amp *= scale_factor
            # >>>
            phase_shift = mean( nr_phi[k]-pn_phi[k] )
            warning('The PN phase will be shifted by %f (rad).'%phase_shift)
            pn_phi += phase_shift
            # >>>
        # --
        # Plotting
        if plot:
            # Import usefuls
            from matplotlib.pyplot import figure,figaspect,plot,xlim,ylim,xscale,subplot,show
            from matplotlib.pyplot import yscale,axvline,axvspan,xlabel,ylabel,title,yscale,legend
            # Create figure
            fig = figure( figsize=3*figaspect(4.0/7) )
            # Plot phases
            subplot(2,1,2)
            if this.aggressive:
                # Hilight the aggressive alignment region
                axvspan( T1k,T2k, color='lawngreen', alpha=0.2 )
                axvline(T1k,color='g',ls='--')
                axvline(T2k,color='g',ls='--')
            plot( t[nr_smoothest_mask], nr_phi[nr_smoothest_mask], color='k', lw=3, alpha=0.4, ls='--' )
            plot( t, nr_phi )
            plot( t, pn_phi, lw=1 )
            # Set plot limits
            ylim( min(pn_phi),max(lim( nr_phi[nr_smoothest_mask], dilate=0.1)) )
            xlim( 0,max(t[nr_smoothest_mask]) )
            # Highlight the hybridization region
            axvspan( T1,T2, alpha=0.15, color='cyan' )
            axvline( T1,color='c',ls='--' )
            axvline( T2,color='c',ls='--' )
            # Set axis labels
            xlabel(r'$t/M$'); ylabel(r'$\phi_{%i%i}=\arg(\psi_{%i%i})$'%(l,m,l,m))
            # Plot amplitudes
            subplot(2,1,1)
            if this.aggressive:
                # Hilight the aggressive alignment region
                axvspan( T1k,T2k, color='lawngreen', alpha=0.2, label='Aggressive alignment region' )
                axvline(T1k,color='g',ls='--')
                axvline(T2k,color='g',ls='--')
            # Add amplitude lines
            yscale('log')
            plot( t[nr_smoothest_mask], nr_amp[nr_smoothest_mask], color='k', lw=3, alpha=0.4, ls='--' )
            plot( t, nr_amp )
            plot( t, pn_amp, lw=1 )
            # Set plot limits
            ylim( min(pn_amp[pn_amp>0]),max(lim( nr_amp[nr_smoothest_mask], dilate=0.1)) )
            xlim( 0,max(t[nr_smoothest_mask]) )
            # Highlight the hybridization region
            axvspan( T1,T2, alpha=0.15, color='cyan', label='Final hybridization region' )
            axvline( T1,color='c',ls='--' )
            axvline( T2,color='c',ls='--' )
            # Set axis labels
            ylabel(r'$|\psi_{%i%i}|$'%(l,m))
            legend()
            # xlabel(r'$t/M$')
            # show()
        # Package output
        foo = {}
        foo['nr_amp'] = nr_amp
        foo['nr_phi'] = nr_phi
        foo['pn_amp'] = pn_amp
        foo['pn_phi'] = pn_phi
        foo['nr_smoothest_mask'] = nr_smoothest_mask
        foo['t'] = t
        # Return aligned phases
        return foo


    # Get the format aligned data
    def __unpacklm__(this,lm,kind):
        '''Get the format aligned data NR and PN data for a single multipole'''
        # Import usefuls
        from numpy import allclose,roll
        # Create shorthand for useful information
        t = this.t
        pno = this.__pn__
        nro = this.gwylmo
        pn_t,pn_y,nr_t,nr_y = pno.pn_gwylmo.t,pno.pn_gwylmo[lm][kind].y,nro.t,nro[lm][kind].y
        # Format align multipole data
        t_,_,nr_y = format_align(t,t*t,nr_t,nr_y,center_domains=True,verbose=False)
        if not allclose(t_,t): error('bad formatting!')
        t_,_,pn_y = format_align(t,t*t,pn_t,pn_y,center_domains=True,verbose=False)
        if not allclose(t_,t): error('bad formatting!')
        # Return answer
        return (t,nr_y,pn_y)


    # Validative constructor for class workflow
    def __validate_inputs__(this,gwylmo,pn_w_orb_min,pn_w_orb_max,kind,aggressive,verbose):

        # Let the people know
        this.verbose = verbose
        if verbose:
            alert('Verbose mode ON.')

        # Toggle for agressiveness of workflow. See doc.
        this.aggressive = aggressive

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
            pn_w_orb_min = 0.8 * strain_w_orb_min
            if this.verbose: alert('Using default values for PN w_orb starting frequency based on strain waveform: %s'%(yellow('%f'%pn_w_orb_min)))
        if pn_w_orb_max is None:
            w_orb_min = gwylmo.wstart_pn/2
            w_orb_merger = gwylmo[2,2]['psi4'].dphi[ gwylmo[2,2]['psi4'].k_amp_max ]/2
            pn_w_orb_max = (w_orb_merger+4*w_orb_min)/5
            if this.verbose: alert('Using default values for PN w_orb end frequency based on strain waveform: %s'%(yellow('%f'%pn_w_orb_max)))

        # Store PN start and end frequency
        this.pn_w_orb_min = pn_w_orb_min
        this.pn_w_orb_max = pn_w_orb_max
        if this.verbose:
            alert('PN w_orb MIN frequency is %s (i.e. w_orb*M_init)'%(yellow('%f'%pn_w_orb_min)))
            alert('PN w_orb MAX frequency is %s (i.e. w_orb*M_init)'%(yellow('%f'%pn_w_orb_max)))

        # Store select inputs
        this.gwylmo = gwylmo
