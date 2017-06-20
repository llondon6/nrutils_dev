# Import useful things
from nrutils.core.basics import *

#
class match:

    #
    def __init__( this,                       # The current object
                  template_wfarr = None,      # column format [ f + x ]
                  signal_wfarr = None,        # column format [ f + x ]
                  psd_name = None,            # determines which psd to use; default is advLigo
                  fmin = 20,
                  fmax = 400,
                  signal_polarization = 0,
                  template_polarization = 0,  # Not used by all types of match calculations
                  input_units = None,
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
                   psd_name = 'aligo' if psd_name is None else psd_name,
                   verbose = verbose)


    #
    def calc_template_phi_optimized_match( this,
                                           template_wfarr_fun, # takes in template orbital phase, outputs waveform array
                                           N_template_phi = 15,# number of orbital phase values to use for exploration
                                           verbose = False ):

        # Import useful things
        from numpy import linspace,pi,array
        from copy import deepcopy as copy
        import matplotlib.pyplot as pp
        that = copy(this)

        #
        if verbose: alert( 'Processing %s over list a phi_template values .'%(cyan('match.calc_template_pol_optimized_match()')), )

        # Define helper function for high level match
        def match_helper(phi_template):
            #
            if verbose: print '.',
            # Get the waveform array at the desired template oribtal phase
            current_template_wfarr = template_wfarr_fun( phi_template )
            # Create the related match object
            that.apply( template_wfarr = current_template_wfarr )
            # Calculate the match
            match = that.calc_template_pol_optimized_match()
            # Return answer
            return match

        # ~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~-- #
        # Estimate optimal match with respect to phi orb of template
        # ~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~-- #

        # Map template orbital phase values to match
        phi_template_range = linspace(0,2*pi,N_template_phi)
        match_list = array( map( match_helper, phi_template_range ) )

        # Interpolate match over phi_template to estimate maximum
        # intrp_max lives in the "positive" repository
        optimal_phi_template = intrp_argmax(match_list,phi_template_range)
        match = match_helper( optimal_phi_template )

        pp.figure()
        pp.plot( phi_template_range, match_list, '-ok', mfc='none', mec = 'k', alpha=0.5 )

        # Return answer
        ans = match
        return ans


    # Plot template and signal against psd
    def plot(this):

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
        figure( figsize=1.8*array([4,4]) )

        #
        def slog(x):
            from numpy import sign,log
            return log(abs(x))*sign(x)

        # Plot
        plot( this.f, 2*sqrt(abs(this.f))*abs(this.signal['response']), label=r'Signal (Response), $\rho_{\mathrm{opt}} = %1.2f$'%this.signal['optimal_snr'] )
        #
        plot( this.f, 2*sqrt(abs(this.f))*abs(this.template['response']), label=r'Template (Response), $\rho_{\mathrm{opt}} = %1.2f$'%this.template['optimal_snr'] )
        #
        plot( this.f, sqrt(this.psd), '-k', label=r'$\sqrt{S_n(f)}$ for '+this.psd_name )

        #
        # xscale('log')
        yscale('log')
        xlim(lim(this.f))
        # ylim( [ sqrt(min(this.psd))/10 , max(ylim()) ] )
        xlabel( r'$f$ (Hz)' )
        ylabel( r'$\sqrt{S_n(f)}$  and  $2|\tilde{h}(f)|\sqrt{f}$' )
        legend( frameon=False, loc=3 )


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
    def calc_template_pol_optimized_match( this, signal_polarization = None ):

        # Import useful things
        from numpy import sqrt,dot,real,log,diff
        from numpy.fft import ifft

        # Handle signal polarization input; use constructor value as default
        if signal_polarization is not None:
            this.signal['polarization']=signal_polarization
            this.__set_waveform_fields__()

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
        num_samples = this.psd.shape[0]
        fftlen = int( 2 ** ( int(log( num_samples )/log(2)) + 1.0 + 3.0 ) )
        df = diff(this.f)[0]
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
        #
        template_pol_optimized_match = sqrt( numerator.max() / (denominator*2.0) ) / (this.signal['norm'] if this.signal['norm'] != 0 else 1)

        # Calculate optimal template polarization angle ?

        #
        ans = template_pol_optimized_match
        return ans


    # Calculate precessing match
    def calc_precessing_match( this, **kwargs ):
        return None


    # Claculate optimal snr
    def calc_optsnr(this,a):
        #
        from numpy import sqrt
        # See Maggiori, p. 345 -- a factor of 2 comes from the definition of the psd
        # One will often see yet another factor of two (making the net factor 4);
        # note that when both m>0 and m<0 multipoles are considered, this cannot be used
        # ALSO see Eqs 1-2 of https://arxiv.org/pdf/0901.1628.pdf
        optsnr = sqrt( 2 * this.calc_overlap(a,method='trapz') )
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
        from numpy import trapz
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
        from numpy import cos,sin
        s = cos(2*psi)*h_plus + sin(2*psi)*h_cross
        # Return the answer
        ans = s
        return ans


    # Set the psd to be used with the current object
    def __set_psd__(this):
        this.__set_psd_fun__()
        # NOTE that we input the absolute value of frequencies per the definition of Sn
        this.psd = this.__psd_fun__( abs(this.f) )


    # Define a function for evaluating the PSD
    def __set_psd_fun__(this):
        '''
        In an if-else sense, the psd_name will be interpreterd as:
        * a short string relating to know psd values
        * the path location of an ascii file containing psd values
        * an array of frequency and psd values
        '''
        # Import useful things
        from numpy import ndarray,loadtxt,sqrt
        import nrutils
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        #
        psd_names_for_tabulated_data = ['aligo','gw150914']
        not_valid_msg = 'unknown psd name found'
        # If psd_name is a string
        psd_name = this.psd_name
        if isinstance(psd_name,str):

            # If psd_name is a standard psd name
            if psd_name.lower() in psd_names_for_tabulated_data:

                # Load the corresponding data
                psd_name = psd_name.lower()
                if 'al' in psd_name:
                    data_path = nrutils.__path__[0]+'/data/ZERO_DET_high_P.dat'
                    psd_arr = loadtxt( data_path )
                    # Validate and unpack the psd array
                    if psd_arr.shape[-1] is 2:
                        psd_f,psd_vals = psd_arr[:,0],psd_arr[:,1]*psd_arr[:,1]
                    else:
                        error('Improperly formatted psd array given. Instead of having two columns, it has %i'%psd_arry.shape[-1])
                elif '091' in psd_name:
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

            elif psd_name.lower() in ('iligo'):

                # use the modeled PSD from nrutils (via Eq. 9 of https://arxiv.org/pdf/0901.1628.pdf)
                psd_fun = lambda f: iligo(f,version=2)

            else:

                error('unknown PSD name: %s'%psd_name)


        elif isinstance(psd_name,ndarray):
            # Else if it's an array of psd data
            psd_arr = psd_name
            # Create an interpolation of the PSD data
            # NOTE that this function is stored to the current object
            psd_fun = spline(psd_f,psd_vals)
        else:
            error(not_valid_msg)

        # Store the PSD function
        this.__psd_fun__ = psd_fun


    # Apply select properties to the current object
    def apply(this,template_wfarr=None, signal_wfarr=None, fmin=None, fmax=None, signal_polarization=None, template_polarization=None, psd_name=None, verbose=None):
        '''
        Apply select attributes to the current object.
        '''

        # Low level handing of inputs
        this.__parse_inputs__(template_wfarr, signal_wfarr, fmin, fmax, signal_polarization, template_polarization, psd_name, verbose)

        # Only reset the psd data if needed
        new_fmin = fmin is not None; new_fmax = fmax is not None
        new_psd = psd_name is not None; reset_psd_data = new_psd or new_fmax or new_fmin
        if reset_psd_data : this.__set_psd__()

        #
        this.__set_waveform_fields__()

        #
        return this


    # Unpack and Validate inputs
    def __parse_inputs__(this, template_wfarr, signal_wfarr, fmin, fmax, signal_polarization, template_polarization, psd_name, verbose):

        # Import useful things
        from numpy import allclose,diff,array,ones

        # Store the verbose input to the current object
        this.verbose = verbose

        #
        if this.verbose:
            alert('Verbose mode on.','match')

        # Store input arrays; this should only been done upon initial construction
        tag = '__input_signal_wfarr__'
        if not ( tag in this.__dict__ ): this.__dict__[tag] = signal_wfarr
        tag = '__input_template_wfarr__'
        if not ( tag in this.__dict__ ): this.__dict__[tag] = template_wfarr

        # Handle optinal inputs; whith previously stored values being defaults in most cases
        this.fmin = fmin if fmin is not None else this.fmin
        this.fmax = fmax if fmax is not None else this.fmax
        # Handle None waveforms arrays
        signal_is_None = signal_wfarr is None
        template_is_None = template_wfarr is None
        if signal_is_None:
            # set signal_wfarr to previously stored value
            signal_wfarr = this.__input_signal_wfarr__
        if template_is_None:
            # set template_wfarr to previously stored value
            template_wfarr = this.__input_template_wfarr__

        # Validate arrays
        this.__validate_wfarrs__(signal_wfarr,template_wfarr)

        # Upack waveform arrays into dictionaries
        this.signal = this.__unpack_wfarr__(signal_wfarr)
        this.template = this.__unpack_wfarr__(template_wfarr)

        # Make common frequncy easily accessible; note that the 'f' key in
        # signal and template has been tested to be all close at this point;
        # these frequencies have also been masked according to fmin and fmax
        this.f = this.signal['f']

        # Store the psd name input
        this.psd_name = psd_name if psd_name is not None else this.psd_name

        # Group pluss and cross into dictionaries
        this.signal['polarization'] = signal_polarization if signal_polarization is not None else 0
        this.template['polarization'] = template_polarization if template_polarization is not None else 0

        '''
        The core functions of this class will operate on this.f, this.signal and this.template
        '''

        #
        return None


    # Store masked wfarr data into dictionary
    def __unpack_wfarr__(this,wfarr):

        #
        f = wfarr[:,0]
        mask = this.__getfmask__( f, this.fmin, this.fmax )

        #
        d = {}
        #
        d['f'] = wfarr[mask,0]
        d['+'] = wfarr[mask,1]
        d['x'] = wfarr[mask,2]

        #
        return d


    # determine mask to be applied to array from fmin and fmax
    def __getfmask__(this,f,fmin,fmax):
        #
        abs_f = abs(f)
        mask = (abs_f>=fmin) & (abs_f<=fmax) & (f!=0)
        #
        return mask


    # Validate the signal_wfarr against template_wfarr
    def __validate_wfarrs__(this,signal_wfarr,template_wfarr):

        # Import useful things
        from numpy import allclose,diff,array,ones

        # arrays should be the same shape
        if signal_wfarr.shape != template_wfarr.shape:
            error('waveform array shapes must be identical')

        # Extract frequencies for comparison
        template_f = template_wfarr[:,0]
        signal_f = signal_wfarr[:,0]

        # Freqs must have same length
        if len(signal_f) != len(template_f):
            print len(signal_f),len(template_f)
            error( 'Frequency columns of waveform arrays are not equal in length. You may wish to interpolate to ensure a common frequency domain space. Please make sure that masking is handled consistently between inputs and possible outputs of recompoase functions (if relevant).' )

        # Freqs must have same values
        if not allclose(signal_f,template_f):
            error("Values in the template frequncy column are not all close to values in the signal frequency column. This should not be the case. Please make sure that masking is handled consistently between inputs and possible outputs of recompoase functions (if relevant).")
