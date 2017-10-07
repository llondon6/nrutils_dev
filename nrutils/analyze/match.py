# Import useful things
from nrutils.core.basics import *
import sys
flush =  sys.stdout.flush

#
class match:

    #
    def __init__( this,                       # The current object
                  template_wfarr = None,      # column format [ f + x ]
                  signal_wfarr = None,        # column format [ f + x ]
                  psd_thing = None,              # determines which psd to use; default is advLigo
                  fmin = 20,
                  fmax = 400,
                  signal_polarization = 0,
                  template_polarization = 0,  # Not used by all types of match calculations
                  input_units = None,
                  positive_f = False, # Toggle for using only positive frequency data; this is automatically set to true of input arrays only contain positive freq data; NOTE that results SHOULD be unchanged when this is true or false
                  verbose = False ):

        '''
        The work of the constructor is delegated to a method so that object attributes can be changed on the fly via this.apply(...)
        '''
        this.apply(template_wfarr = template_wfarr,
                   signal_wfarr = signal_wfarr,
                   fmin = fmin,
                   fmax = fmax,
                   signal_polarization = signal_polarization,
                   template_polarization = template_polarization,
                   psd_thing = 'aligo' if psd_thing is None else psd_thing,
                   positive_f = positive_f,
                   verbose = verbose)


    #
    def calc_template_phi_optimized_match( this,
                                           template_wfarr_orbphi_fun, # takes in template orbital phase, outputs waveform array
                                           N_template_phi = 61,# number of orbital phase values to use for exploration
                                           plot = False,
                                           signal_polarization = None,
                                           method = None,
                                           verbose = False,
                                           **kwargs ):

        # Import useful things
        from numpy import linspace,pi,array,argmax,isnan
        import matplotlib.pyplot as pp

        #
        if verbose: alert( 'Processing %s over list a phi_template values.'%(cyan('match.calc_template_pol_optimized_match()')), )


        # Handle signal polarization input; use constructor value as default
        if signal_polarization is not None:
            this.apply( signal_polarization=signal_polarization )

        #
        if method is None:
            method = 'analytic'
        elif not isinstance(method,str):
            error('method must be string')
        else:
            method = method.lower()

        #
        if method in ('numerical'):
            matchfun = this.brute_match
        elif method in ('analytic'):
            matchfun = this.calc_template_pol_optimized_match
        else:
            error('unknown method for calculating template polarization optimised match: %s'%method)


        # Define helper function for high level match
        def match_helper(phi_template):
            #
            if verbose: print '.',
            # Get the waveform array at the desired template oribtal phase
            current_template_wfarr = template_wfarr_orbphi_fun( phi_template )
            # Create the related match object
            this.apply( template_wfarr = current_template_wfarr )
            # Calculate the match
            match = matchfun(**kwargs)
            #
            if isnan(match):
                error('The match should not be %s. There\'s a bug to be fixed here.'%match)
            # Return answer
            return match

        # ~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~-- #
        # Estimate optimal match with respect to phi orb of template
        # ~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~-- #

        # Map template orbital phase values to match
        phi_template_range = linspace(0,2*pi,N_template_phi)
        match_list = abs( array( map( match_helper, phi_template_range ) ) )

        # Interpolate match over phi_template to estimate maximum
        # intrp_max lives in the "positive" repository
        optimal_phi_template = intrp_argmax(match_list,phi_template_range,plot=plot)
        match = max( match_helper( optimal_phi_template ), max(match_list) )

        #
        if plot:
            from matplotlib import pyplot as pp
            pp.figure()
            pp.plot( phi_template_range, match_list, '-ob', mfc='none', mec = 'k' )
            pp.xlim( lim(phi_template_range) )
            pp.xlabel(r'$\phi_{\mathrm{template}}$ (Template Orbital Phase)')
            pp.ylabel(r'$ \operatorname{max}_{t,\phi_{\mathrm{arrival}},\psi_{\mathrm{template}}} \langle s | h \rangle$')
            pp.title('max = %f'%match )
            pp.show()

        # Return answer
        ans = match
        return ans


    # Estimate the min, mean, and max match when varying signal polarization and orbital phase
    def calc_match_sky_moments( this,
                                signal_wfarr_fun,   # takes in template inclination & orbital phase, outputs waveform array
                                template_wfarr_fun, # takes in signal   inclination & orbital phase, outputs waveform array
                                N_theta = 25,
                                N_psi_signal = 13,
                                N_phi_signal = 13,
                                hm_vs_quad = False,
                                plot = False,
                                method = None,
                                verbose = not False,
                                **kwargs ):

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        # Import useful things
        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        from numpy import array,linspace,pi,mean,average,zeros,pi
        if plot:
            from numpy import meshgrid
            import matplotlib.pyplot as pp
            from mpl_toolkits.mplot3d import Axes3D

        #
        if hm_vs_quad:
            alert('We will also compute matches for the quadrupole only template.')

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        # Define 3D grid
        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        # Inclination
        et = 0.01
        theta_range = linspace(et,pi-et,N_theta) if N_theta > 1 else array([pi/2])
        # theta_range = linspace(0,pi,N_theta)
        # Signal polarization
        psi_signal_range = linspace(et,pi-et,N_psi_signal)
        # Signal orbital phase
        phi_signal_range = linspace(et,2*pi-et,N_phi_signal)

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        # Evaluate match over grid
        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#

        # initialize lists
        min_match,avg_match,snr_avg_match,max_match = [],[],[],[]
        quadrupole_min_match = []
        quadrupole_avg_match = []
        quadrupole_snr_avg_match = []
        quadrupole_max_match = []
        #
        samples = {'theta':[],'phi_signal':[],'psi_signal':[],'match':[],'optsnr':[]}
        quadrupole_samples = {'theta':[],'phi_signal':[],'psi_signal':[],'match':[],'optsnr':[]}

        '''Loop over INCLINATION'''
        for theta in theta_range:

            # Let the people know
            if verbose: alert('theta = %1.2f\n%s'%(theta,20*'--'))

            # Construct a representation of the template waveform evaluated at the current value of theta
            template_fun = lambda PHI: template_wfarr_fun(theta,PHI,None)
            # NOTE that in the line above, None MUST default to all available LM multipoles being used
            quadrupole_lm = [(2,2),(2,-2)]
            quadrupole_template_fun = lambda PHI: template_wfarr_fun(theta,PHI,quadrupole_lm)

            #------------------------#
            # For all reference orbital phase angles, calculate matches and snrs
            #------------------------#

            #
            match_list,optsnr_list = [],[]
            quadrupole_match_list = []

            #
            match_arr = zeros( (N_phi_signal,N_psi_signal), dtype=float )

            #
            if verbose:
                print '>> working ',
                flush()

            '''Loop over SIGNAL ORBITAL PHASE'''
            for j,phi_signal in enumerate(phi_signal_range):

                # Evaluate the signal representation at this orbital phase
                signal_wfarr = signal_wfarr_fun(theta,phi_signal)
                this.apply( signal_wfarr=signal_wfarr )

                '''Loop over SIGNAL POLARIZATION'''
                for k,psi_signal in enumerate(psi_signal_range):
                    print '.',
                    flush()
                    # Apply the signal poliarzation to the current object
                    this.apply( signal_polarization = psi_signal )
                    #
                    '''
                    Optimize match over TEMPLATE ORBITAL PHASE and
                    template phase, polarization and time
                    '''
                    optsnr = this.signal['optimal_snr']

                    #
                    if hm_vs_quad:
                        quadrupole_match = this.calc_template_phi_optimized_match(quadrupole_template_fun,method=method)
                        quadrupole_match_list.append( quadrupole_match )
                    #
                    match = this.calc_template_phi_optimized_match(template_fun,method=method)
                    match_list.append( match )
                    match_arr[j,k] = match
                    #
                    optsnr_list.append( optsnr )
                    #
                    samples['theta'].append(theta)
                    samples['phi_signal'].append(phi_signal)
                    samples['psi_signal'].append(psi_signal)
                    samples['match'].append(match)
                    samples['optsnr'].append(optsnr)
                    #
                    if hm_vs_quad:
                        #
                        quadrupole_samples['theta'].append(theta)
                        quadrupole_samples['phi_signal'].append(phi_signal)
                        quadrupole_samples['psi_signal'].append(psi_signal)
                        quadrupole_samples['match'].append(quadrupole_match)
                        quadrupole_samples['optsnr'].append(optsnr)

                # For all signal polarization values
                if verbose:
                    print ',',
                    flush()

                #
                if plot:
                    pp.figure()
                    pp.plot( psi_signal_range, match_arr[j,:], '-o' )
                    pp.title( r'$\theta = %f, \phi_{\mathrm{signal}} = %f$'%(theta,phi_signal) )
                    pp.xlabel(r'$\psi_\mathrm{signal}$')
                    pp.show()

            #------------------------#
            # Calculate moments for this inclination
            #------------------------#
            if verbose:
                print ' done.'
                flush()
            # convert to arrays to help with math
            match_list,optsnr_list = array(match_list),array(optsnr_list)
            quadrupole_match_list = array(quadrupole_match_list)
            # calculate min mean and max (no snr weighting)
            min_match.append(  min(match_list) )
            max_match.append(  max(match_list) )
            avg_match.append( mean(match_list) )
            #
            if hm_vs_quad:
                quadrupole_min_match.append(  min(quadrupole_match_list) )
                quadrupole_max_match.append(  max(quadrupole_match_list) )
                quadrupole_avg_match.append( mean(quadrupole_match_list) )
            # calculate min mean and max (with SIGNAL snr weighting)
            snr_avg_match.append( average( match_list, weights=optsnr_list**3 ) )
            if hm_vs_quad:
                quadrupole_snr_avg_match.append( average( quadrupole_match_list, weights=optsnr_list**3 ) )
            #
            print '>>  min_match \t = \t %f' % min_match[-1]
            print '>>  avg_match \t = \t %f' % avg_match[-1]
            print 'snr_avg_match \t = \t %f' % snr_avg_match[-1]
            print '>>  max_match \t = \t %f' % max_match[-1]
            if hm_vs_quad:
                print '##  quadrupole_min_match \t = \t %f' % quadrupole_min_match[-1]
                print '##  quadrupole_avg_match \t = \t %f' % quadrupole_avg_match[-1]
                print 'quadrupole_snr_avg_match \t = \t %f' % quadrupole_snr_avg_match[-1]
                print '##  quadrupole_max_match \t = \t %f' % quadrupole_max_match[-1]
            print '--'*20
            flush()

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        # Store moments to dictionary for output
        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
        match_info = { 'theta':theta_range }
        #
        match_info['min'] = min_match
        match_info['avg'] = avg_match
        match_info['max'] = max_match
        match_info['weighted_avg'] = snr_avg_match
        match_info['samples'] = samples
        #
        if hm_vs_quad:
            match_info['quadrupole_min'] = quadrupole_min_match
            match_info['quadrupole_avg'] = quadrupole_avg_match
            match_info['quadrupole_max'] = quadrupole_max_match
            match_info['quadrupole_weighted_avg'] = quadrupole_snr_avg_match
            match_info['quadrupole_samples'] = quadrupole_samples

        #
        return match_info


    #
    def franks_match( this ):

        #
        from numpy import angle,sqrt,exp,log,arctan,sin,cos
        from numpy.fft import ifft

        #
        error('the code here is still in development!')

        #
        h1dat = this.template['+'] + 1j*this.template['x']
        h2dat = this.signal['+'] + 1j*this.signal['x']

        #
        n1fun = lambda x: sum( abs(x)**2   / this.psd )
        n2fun = lambda x: sum( x * x[::-1] / this.psd )
        norm11,norm21 = n1fun(h1dat),n1fun(h2dat)
        norm12,norm22 = n2fun(h1dat),n2fun(h2dat)
        N2 = abs(norm22); sigma1 = angle(norm22)
        print norm11, norm21
        print abs(norm12), abs(norm22)

        #
        int1 = h1dat * h2dat.conj() / this.psd
        int2 = h1dat.conj()[::-1] * h2dat.conj() / this.psd
        num_samples = len(int2)
        fftlen = int( 2 ** ( int(log( num_samples )/log(2)) + 1.0 + 2.0 ) )

        #
        FFTs = [ ifft(int1,n=fftlen)*fftlen, ifft(int2,n=fftlen)*fftlen ]

        # Content of alloptmatches
        def alloptmatches(psi):
            #
            h1norm = sqrt( norm11 + (norm12*exp(1j*4*psi)).real  )
            #
            sumFFTs = FFTs[0]*exp( 1j*2*psi ) + FFTs[1]*exp( -1j*2*psi )
            #
            absMatch = abs(sumFFTs); sigma2 = angle( sumFFTs )
            # from matplotlib import pyplot as pp
            # pp.figure()
            # pp.plot( absMatch )
            # pp.show()
            #
            polOptMatches = absMatch * sqrt(  ((norm21**2)-(N2**2))*(norm21-N2*cos(sigma1+2*sigma2))  ) / ( ((norm21**2)-(N2**2))*h1norm )
            #
            optMatch = max( polOptMatches )
            # from matplotlib import pyplot as pp
            # pp.figure()
            # pp.plot( absMatch )
            # pp.show()

            #
            ## Not needed:
            # besttime = argmax(polOptMatches)
            # sigma2opt = sigma2[besttime];
            # optpol = arctan2( norm21*sin(sigma2opt)+N2*sin(sigma1+sigma2opt) , norm21*cos(sigma2opt)-N2*cos(sigma1+sigma2opt) ) / 2
            #
            return optMatch

        #
        ans = alloptmatches
        return ans


    # basic match function that optimizes over template polarization in a brute force way
    def brute_match( this, **kwargs ):

        #
        from numpy import pi,linspace,argmax
        from scipy.interpolate import InterpolatedUnivariateSpline as spline

        # define range of template polarizations to consider for loop
        template_polarization_range = linspace( 0, pi, 29 )
        # store the current value
        default_template_polarization = this.template['polarization']

        # define list to store temp match values
        match_list = []

        #
        matchfun = lambda PSI: this.calc_basic_match( template_polarization = PSI, **kwargs )

        # for all template polarization values, calculate basic match which optimizes over arrival phase, and time shift
        for current_template_polarization in template_polarization_range :

            # Given the current template polarization, calculate the basic match
            # this match maximizes over time offset and
            basic_match = matchfun( current_template_polarization )
            match_list.append( basic_match )

        # Estimate the maximum match
        roots = spline(template_polarization_range,match_list,k=4).derivative().roots()
        max_matches = map( matchfun, roots )
        best_polarization = template_polarization_range[argmax( max_matches )]
        best_match = max( max_matches )

        #
        ans = best_match
        return ans

    #
    def calc_basic_match( this,
                          ifftmax = True,       # Toggel for time maximization via ifft
                          template_polarization = None,
                          timeoptoff = False,   # TUrn off time shift optimization
                          verbose = False ):

        #
        from numpy import log,exp
        from numpy.fft import ifft

        # Apply the template polarization; NOTE that we do not use the apply method here as it is too general for our cause. NOTE that one is always free to use the apply method externally
        if template_polarization is None:
            template_polarization = this.template['polarization']
        else:
            this.template['polarization'] = template_polarization
            this.__set_waveform_fields__()

        #
        G = this.signal['normalized_response'] * this.template['normalized_response'].conj() / this.psd

        #
        if not timeoptoff:
            fftlen = int( 2 ** ( int(log( len( this.psd ) )/log(2)) + 1.0 + 3.0 ) )
            H = ifft( G, n = fftlen )*fftlen
            ans = max( abs(H) )
        else:
            ans = sum( G )

        # Optimize over phase
        match = abs(ans)

        # Return the answer
        return match


    # Plot template and signal against psd
    def plot(this,fig=None):

        # Import useful things
        from numpy import array,sqrt,log,sign

        # Setup plotting backend
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 16
        mpl.rcParams['axes.titlesize'] = 16
        from matplotlib.pyplot import plot,show,xlabel,ylabel,title,\
                                      legend,xscale,yscale,xlim,ylim,figure

        # Setup Figure
        if fig is None:
            fig = figure( figsize=3*array([4,4]) )

        #
        def slog(x):
            from numpy import sign,log
            return log(abs(x))*sign(x)

        # Plot
        plot( this.f, 2*sqrt(abs(this.f))*abs(this.signal['response']), label=r'Signal (Response), $\rho_{\mathrm{opt}} = %1.2f$'%this.signal['optimal_snr'] )
        #
        plot( this.f, 2*sqrt(abs(this.f))*abs(this.template['response']), label=r'Template (Response), $\rho_{\mathrm{opt}} = %1.2f$'%this.template['optimal_snr'] )
        #
        if callable(this.psd_thing):
            psd_label = this.psd_thing.__name__
        else:
            psd_label = str(this.psd_thing)
        plot( this.f, sqrt(this.psd), '-k', label=r'$\sqrt{S_n(f)}$ for %s'%psd_label )

        #
        yscale('log')
        a = 2*sqrt(abs(this.f))*abs(this.template['response'])
        a = a[ (abs(a)>0) & (abs(this.f)>this.fmin) ]
        yl = lim(a)
        ylim( yl )

        #
        xlim(lim(this.f))
        xscale('log')
        xlabel( r'$f$ (Hz)' )
        ylabel( r'$\sqrt{S_n(f)}$  and  $2|\tilde{h}(f)|\sqrt{f}$' )
        legend( frameon=False, loc=1 )


    # Precalculate items related to the signals
    def __set_waveform_fields__(this):

        # NOTE that some fields are set in the __parse_inputs__ method

        # Import useful things
        from numpy import sqrt

        # Calculate the combination of SIGNAL + and x as determined by the signal_polarization
        this.signal['response'] = this.calc_eff_detector_response( this.signal['polarization'], this.signal['+'], this.signal['x'] )

        # Calculate the combination of TEMPLATE + and x as determined by the signal['polarization']
        this.template['response'] = this.calc_eff_detector_response( this.template['polarization'], this.template['+'], this.template['x'] )

        # Calculate the normalizatino constant to be used for DFT related matches
        # NOTE that inconsistencies here in the method used can affect consistency of match
        this.signal['norm'] = this.calc_norm( this.signal['response'], method='sum' )
        this.template['norm'] = this.calc_norm( this.template['response'], method='sum' )

        # Store Optimal SNRs (e.g. Eq 3.6 of https://arxiv.org/pdf/gr-qc/0604037.pdf)
        this.signal['optimal_snr']   = this.calc_optsnr( this.signal['response'] )
        this.template['optimal_snr'] = this.calc_optsnr( this.template['response'] )

        # Calculate the normed signal with respect to the overlap
        this.signal['normalized_response'] = this.signal['response'] / (this.signal['norm'] if this.signal['norm'] != 0 else 1)

        # Calculate the normed template with respect to the overlap
        this.template['normalized_response'] = this.template['response'] / (this.template['norm'] if this.template['norm'] != 0 else 1)

        return None


    # Calculate overlap optimized over template polarization
    def calc_template_pol_optimized_match( this, signal_polarization = None, plot = False ):

        # Import useful things
        from numpy import sqrt,dot,real,log,diff
        from numpy.fft import ifft

        # Handle signal polarization input; use constructor value as default
        if signal_polarization is not None:
            this.apply( signal_polarization=signal_polarization )

        #
        this.template['+norm'] = this.calc_norm( this.template['+'] )
        this.template['xnorm'] = this.calc_norm( this.template['x'] )
        #
        if (this.template['+norm']==0) or (this.template['xnorm']==0) :
            print sum(abs(this.template['+']))
            print sum(abs(this.template['x']))
            print 'template +norma = %f'%this.template['+norm']
            print 'template xnorma = %f'%this.template['xnorm']
            error('Neither + nor x of template can be zero for all frequencies.')
        #
        normalized_template_plus  = this.template['+']/this.template['+norm']
        normalized_template_cross = this.template['x']/this.template['xnorm']
        IPC = real( this.calc_overlap(normalized_template_plus,normalized_template_cross) )

        # Padding for effective sinc interpolation of ifft
        num_samples = len(this.psd)
        fftlen = int( 2 ** ( int(log( num_samples )/log(2)) + 1.0 + 3.0 ) )
        # fftlen = num_samples
        #
        integrand = lambda a: a.conj() * this.signal['response'] / this.psd
        rho_p = ifft( integrand(normalized_template_plus ) , n=fftlen )*fftlen
        rho_c = ifft( integrand(normalized_template_cross) , n=fftlen )*fftlen

        #
        gamma = ( rho_p * rho_c.conj() ).real
        rho_p2 = abs( rho_p ) ** 2
        rho_c2 = abs( rho_c ) ** 2
        sqrt_part = sqrt( (rho_p2-rho_c2)**2 + 4*(IPC*rho_p2-gamma)*(IPC*rho_c2-gamma) )
        numerator = rho_p2 - 2.0*IPC*gamma + rho_c2 + sqrt_part
        denominator = 1.0 - IPC**2

        # plotting
        if plot:
            #
            from matplotlib.pyplot import figure,plot,show,xlabel,ylabel,xlim\
            ,ylim,yscale,xscale,title,legend
            from numpy import argmax,array
            #
            figure()
            plot( abs(integrand(normalized_template_plus )) )
            #
            figure()
            M = sqrt( numerator / (denominator*2.0) ) / (this.signal['norm'] if this.signal['norm'] != 0 else 1)
            plot( M )
            xlim( argmax(abs(rho_p)) + 200*array([-1,1]) )
            show()
            #
            intrp_max( M, plot=True, verbose=True )

        #
        # print '\n',denominator, this.signal['norm'], this.template['norm']
        template_pol_optimized_match = sqrt( numerator.max() / (denominator*2.0) ) / (this.signal['norm'] if this.signal['norm'] != 0 else 1)

        # Calculate optimal template polarization angle ?

        #
        ans = template_pol_optimized_match
        return ans


    # Claculate optimal snr
    def calc_optsnr(this,a):
        #
        from numpy import sqrt
        # See Maggiori, p. 345 -- a factor of 2 comes from the definition of the psd
        # another factor of 2 is picked up if only positive frequencies are used for the integral; this is due to the signal being real valued and thus having identical power at positive and negative frequencies
        # ALSO see Eqs 1-2 of https://arxiv.org/pdf/0901.1628.pdf
        optsnr = sqrt( (4 if this.__positive_f__ else 2) * this.calc_overlap(a,method='trapz') )
        return optsnr


    # Calculate noise curve weighted norm (aka Optimal SNR)
    def calc_norm(this,a,method=None):
        #
        from numpy import sqrt
        #
        norm = sqrt( this.calc_overlap(a,method=method) )
        #
        ans = norm
        return ans


    # Calculate the PSD weighted inner-product between to waveforms
    def calc_overlap(this,a,b=None,method=None):
        #
        from numpy import trapz,isinf,isnan
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        from matplotlib import pyplot as pp
        #
        b = a if b is None else b
        #
        if (this.psd.shape[0] != b.shape[0]) or (this.psd.shape[0] != a.shape[0]):
            print this.psd.shape, a.shape, b.shape
            error('vector shapes not compatible with this objects psd shape')
        #
        integrand = ( abs(a)**2 if b is a else a.conj()*b ) / this.psd
        # print '\n',max(this.psd)
        # print max(abs(a)**2)
        # print max(integrand)
        # pp.figure()
        # pp.plot( this.f, integrand )
        #
        default_method = 'sum' # 'trapz'
        method = default_method if method is None or not isinstance(method,str) else method
        #
        if not method in ('trapz','spline','sum'): error('unknown integration method input; must be spline or trapz')
        #
        if method in ('trapz'):
            overlap = trapz(integrand,this.f)
        elif method in ('spline'):
            overlap = spline(this.f,integrand).integral(this.fmin,this.fmax)
        elif method in ('sum'):
            overlap = sum( integrand )
        #
        ans = overlap
        return ans


    # Combine singal component polarizations using a given effective polarization angle (NOTE that this is not the same polarization angle used in a specific antenna response, but some effective representation of it)
    def calc_eff_detector_response(this,psi,h_plus,h_cross):
        '''Here we interpret PSI to be an effective polarization angle that
        encompasses the effect of sky location, antenna pattern, and wave-frame
        signal polarization. The factor of two is not really necessary?'''

        # Import useful things
        from numpy import cos,sin,pi

        #
        if psi in (pi/4,3*pi/4):
            # warning('Polarization value near pi/4 or 3*pi/4 given. We will shift the input polarization value by some very small amount to avoid spurious results.')
            psi += 0.00001

        #
        s = cos(2*psi)*h_plus + sin(2*psi)*h_cross

        # Return the answer
        ans = s
        return ans


    # Set the psd to be used with the current object
    def __set_psd__(this):
        this.__set_psd_fun__()
        # NOTE that we input the absolute value of frequencies per the definition of Sn
        from numpy import array
        this.psd = array( map(this.__psd_fun__,abs(this.f)) )
        # this.__psd_fun__( abs(this.f) )


    # Define a function for evaluating the PSD
    def __set_psd_fun__(this):
        '''
        In an if-else sense, the psd_thing will be interpreterd as:
        * a short string relating to know psd values
        * the path location of an ascii file containing psd values
        * an array of frequency and psd values
        '''
        # Import useful things
        from numpy import ndarray,loadtxt,sqrt
        #
        psd_things_for_tabulated_data = ['aligo','gw150914']
        not_valid_msg = 'unknown psd name found'

        # #
        # try:
        #     from lalsimulation import SimNoisePSDaLIGOZeroDetHighPower as aligo
        # except:
        #


        '''If psd_thing is a string'''
        psd_thing = this.psd_thing
        if isinstance(psd_thing,str):

            # If psd_thing is a standard psd name
            if psd_thing.lower() in psd_things_for_tabulated_data:

                # Load the corresponding data
                psd_thing = psd_thing.lower()
                if 'al' in psd_thing:
                    data_path = nrutils.__path__[0]+'/data/ZERO_DET_high_P.dat'
                    psd_arr = loadtxt( data_path )
                    # Validate and unpack the psd array
                    if psd_arr.shape[-1] is 2:
                        psd_f,psd_vals = psd_arr[:,0],psd_arr[:,1]*psd_arr[:,1]
                    else:
                        error('Improperly formatted psd array given. Instead of having two columns, it has %i'%psd_arry.shape[-1])
                elif '091' in psd_thing:
                    data_path = nrutils.__path__[0]+'/data/H1L1-AVERAGE_PSD-1127271617-1027800.txt'
                    psd_arr = loadtxt( data_path )
                    # Validate and unpack the psd array
                    if psd_arr.shape[-1] is 2:
                        psd_f,psd_vals = psd_arr[:,0],psd_arr[:,1]
                    else:
                        error('Improperly formatted psd array given. Instead of having two columns, it has %i'%psd_arry.shape[-1])

                # Create an interpolation of the PSD data
                # NOTE that this function is stored to the current object
                psd_fun = spline(psd_f,psd_vals)

            elif psd_thing.lower() in ('iligo'):

                # use the modeled PSD from nrutils (via Eq. 9 of https://arxiv.org/pdf/0901.1628.pdf)
                psd_fun = lambda f: iligo(f,version=2)

            else:

                error('unknown PSD name: %s'%psd_thing)


        elif isinstance(psd_thing,ndarray):
            '''Else if it's an array of psd data'''
            psd_arr = psd_thing
            # Create an interpolation of the PSD data
            # NOTE that this function is stored to the current object
            psd_fun = spline(psd_f,psd_vals)
        elif callable(psd_thing):
            '''Else if it's a function'''
            psd_fun = psd_thing
        else:
            error(not_valid_msg)

        # Store the PSD function
        this.__psd_fun__ = psd_fun


    # Apply select properties to the current object
    def apply(this,template_wfarr=None, signal_wfarr=None, fmin=None, fmax=None, signal_polarization=None, template_polarization=None, psd_thing=None, positive_f=None, verbose=None):
        '''
        Apply select attributes to the current object.
        '''

        # Low level handing of inputs
        this.__parse_inputs__(template_wfarr, signal_wfarr, fmin, fmax, signal_polarization, template_polarization, psd_thing, positive_f, verbose)

        # Only reset the psd data if needed
        new_fmin = fmin is not None; new_fmax = fmax is not None
        new_psd = psd_thing is not None; reset_psd_data = new_psd or new_fmax or new_fmin
        if reset_psd_data : this.__set_psd__()

        #
        this.__set_waveform_fields__()

        #
        return this


    # Unpack and Validate inputs
    def __parse_inputs__(this, template_wfarr, signal_wfarr, fmin, fmax, signal_polarization, template_polarization, psd_thing, positive_f, verbose):

        # Import useful things
        from numpy import allclose,diff,array,ones

        # Store the verbose input to the current object
        this.verbose = verbose

        #
        if this.verbose:
            alert('Verbose mode on.','match')

        # Handle optinal inputs; whith previously stored values being defaults in most cases
        this.fmin = fmin if fmin is not None else this.fmin
        this.fmax = fmax if fmax is not None else this.fmax

        #-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~#
        # Handle None waveforms arrays
        #-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~#
        # SIGNAL
        if signal_wfarr is None:
            # set signal_wfarr to previously stored value
            signal_wfarr = this.__input_signal_wfarr__
        else:
            # Use the input value to set the default value for this object
            this.__input_signal_wfarr__ = signal_wfarr
        # TEMPLATE
        if template_wfarr is None:
            # set template_wfarr to previously stored value
            template_wfarr = this.__input_template_wfarr__
        else:
            # Use the input value to set the default value for this object
            this.__input_template_wfarr__ = template_wfarr

        #-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~#
        # Determine how to handle sidedness of data
        #-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~#
        if positive_f is None:
            if '__positive_f__' in this.__dict__:
                positive_f = this.__positive_f__
            else:
                positive_f = (min(signal_wfarr[:,0].real)>0) and (min(template_wfarr[:,0].real)>0)
        #
        this.__positive_f__ = positive_f

        # Upack waveform arrays into dictionaries
        this.signal = this.__unpack_wfarr__(signal_wfarr, this.signal if 'signal' in this.__dict__ else None )
        this.template = this.__unpack_wfarr__(template_wfarr, this.template if 'template' in this.__dict__ else None )

        # Validate arrays // NOTE that this method acts on this.signal and this.template
        this.__validate_wfarrs__()

        # Make common frequncy easily accessible; note that the 'f' key in
        # signal and template has been tested to be all close at this point;
        # these frequencies have also been masked according to fmin and fmax
        this.f = this.signal['f']

        # Store the psd name input
        this.psd_thing = psd_thing if psd_thing is not None else this.psd_thing

        # NOTE the following polarization value cases:
        # * 0 if the object is initializing and None is given
        # * no value change of None is given and the object was previously initialized
        # * the new value if it is not None

        if signal_polarization is None:
            #
            # print this.signal.keys()
            signal_polarization = this.signal['polarization']
        #
        this.signal['polarization'] = signal_polarization

        # this.signal['polarization'] = signal_polarization if signal_polarization is not None else ( 0 if 'polarization' not in this.signal else this.signal['polarization'] )

        this.template['polarization'] = template_polarization if template_polarization is not None else ( 0 if 'polarization' not in this.template else this.template['polarization'] )

        '''
        The core functions of this class will operate on this.f, this.signal and this.template
        '''

        #
        return None


    # Store masked wfarr data into dictionary; NOTE that the 3rd input is a seed dicionary
    def __unpack_wfarr__(this,wfarr,d=None):

        #
        from numpy import diff,var,mod,array,arange,argmin,argmax
        from numpy.fft import fftshift,ifftshift

        # Detect fft convention
        def monotonic_convention(_wfarr):
            f = _wfarr[:,0]
            return (argmin(f)==0) and (argmax(f)==len(f[1:]))
        #
        if not monotonic_convention(wfarr):
            wfarr = ifftshift( wfarr, 0 ) if mod(len(wfarr[:,0]),2) else fftshift( wfarr, 0 )
            f = array( wfarr[:,0] )
            space = arange( wfarr.shape[0] )
            chart = sorted( space, key = lambda k: f[k] )
            wfarr = wfarr[ chart, : ]
            if this.verbose: alert(yellow('Trying to flip FFT convention so that frequencies are monotocinally increasing from right to left in arrays.'))
            if not monotonic_convention(wfarr):
                error('Attempt to switch convention failed.')

        # Replace zero frequency value with a very small number
        f = wfarr[:,0].real
        abs_f = abs(f)
        if 0 in f:
            smallest_f = min( abs_f[ abs_f>0 ] )
            wfarr[ f==0, 0 ] = 1e-6 * smallest_f

        # Remove undesired frequency conetnt according to fmax and fmin:
        # * Strain values between -fmin and +fmin are set to zero
        # * All data above +-fmax are cropped away
        filtered_wfarr = this.__filter_wfarr__( wfarr, this.fmin, this.fmax )

        #
        d = {} if d is None else d
        #
        d['f'] = filtered_wfarr[:,0].real # wfarr is complex typed, thus the freq vals have a 0*1j imag part that needs to be removed
        d['+'] = filtered_wfarr[:,1]
        d['x'] = filtered_wfarr[:,2]

        # Validate frequency array
        if var(diff( d['f'] )) > 1e-6 :
            from matplotlib import pyplot as pp
            pp.figure()
            pp.plot( d['f'] )
            pp.show()
            error('Frequency series must be uniformly spaced; otherwise time shift optimization via ifft will not work correctly. Please inspect the input frequency series.')

        #
        return d


    # method for formatting array; this is only useful if positive and negative frequency data is used in the  workflow
    def __filter_wfarr__( this, wfarr, fmin, fmax ):

        # Import useful things
        from numpy import ones_like

        # work with a copy
        wfarr = wfarr.copy()

        #
        if this.__positive_f__ :

            #
            f = wfarr[:,0].real
            mask = ( f >= fmin ) & ( f <= fmax )

            #
            wfarr = wfarr[ mask, : ]

        else:

            #
            f = wfarr[:,0].real
            abs_f = abs(f)

            # Zero away low freq data
            wfarr[ abs_f < fmin, 1: ] *= 0

            # Crop high frequency data
            mask = abs_f<=fmax
            wfarr = wfarr[ mask, : ]

        #
        ans = wfarr
        return ans



    # Check conjugate symmetry of FD series
    def __check_conjugate_sym__(this,f,vals):

        #
        from numpy import allclose

        #
        j = f >  this.fmin
        k = f < -this.fmin

        #
        symmetry_broken = not allclose( vals[j], vals[k][::-1].conj(), rtol=1e-3 )
        # print symmetry_broken
        if symmetry_broken:
            #
            msg = '''
                Double sided frequency spectra input, but the necessary conjugate symmetry for
                at least one of the poalrizations is found to be broken. Each FD polarization
                must the be fft of h+ or hx, where each is the real valued time domain waveform.
                The fourier transform of real valued series, T(f), must obey: T(f) = Conjugate( T(-f) ).
            '''
            #
            error(msg)


    # Validate the signal_wfarr against template_wfarr
    def __validate_wfarrs__(this):

        # Import useful things
        from numpy import allclose,diff,array,ones,vstack,allclose

        #
        signal_wfarr   = vstack( [ this.signal['f'],this.signal['+'],this.signal['x'] ] ).T
        template_wfarr = vstack( [ this.template['f'],this.template['+'],this.template['x'] ] ).T

        # arrays should be the same shape
        if signal_wfarr.shape != template_wfarr.shape:
            message = '''

            Waveform array shapes must be identical:

            Signal shape:   %s
            Template shape: %s
            Signal df:      %f
            Template df:    %f

            '''%(signal_wfarr.shape,template_wfarr.shape,diff(signal_wfarr[:,0])[0],diff(template_wfarr[:,0])[0])
            error(message)

        # Extract frequencies for comparison
        template_f = template_wfarr[:,0]
        signal_f = signal_wfarr[:,0]

        # If there are negative frequencies, make sure that inputs obey conjugate symmetry
        if not this.__positive_f__:
            for sym in ['+','x']:
                this.__check_conjugate_sym__( this.template['f'], this.template[sym] )
                this.__check_conjugate_sym__( this.signal['f'], this.signal[sym] )

        # Freqs must have same length
        if len(signal_f) != len(template_f):
            print len(signal_f),len(template_f)
            error( 'Frequency columns of waveform arrays are not equal in length. You may wish to interpolate to ensure a common frequency domain space. Please make sure that masking is handled consistently between inputs and possible outputs of recompoase functions (if relevant).' )

        # Freqs must have same values
        if not allclose(signal_f,template_f):
            error("Values in the template frequncy column are not all close to values in the signal frequency column. This should not be the case. Please make sure that masking is handled consistently between inputs and possible outputs of recompoase functions (if relevant).")
