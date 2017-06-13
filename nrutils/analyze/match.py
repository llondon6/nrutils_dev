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
                  verbose = True ):

        '''
        The work of the constructor is delegated to a method so that object attributes can be changed on the fly via this.apply(...)
        '''
        this.apply(template_wfarr = template_wfarr,
                   signal_wfarr = signal_wfarr,
                   fmin = fmin,
                   fmax = fmax,
                   signal_polarization = signal_polarization,
                   template_polarization = template_polarization,
                   psd_name = psd_name,
                   verbose = verbose)


    # Plot template and signal against psd
    def plot(this):
        # Import useful things
        from numpy import array,sqrt
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
        figure( figsize=1.8*array([4.5,4]) )
        # Plot
        plot( this.f, this.psd, '-k', label=this.psd_name )
        plot( this.f, 2*sqrt(this.f)*abs(this.signal['response']), label=r'Signal, $\rho_{\mathrm{opt}} = %1.2f$'%this.signal['optimal_snr'] )
        print this.signal['optimal_snr']
        plot( this.f, 2*sqrt(this.f)*abs(this.template['response']), label='Template' )
        #
        xscale('log'); yscale('log')
        xlim(lim(this.f))
        ylim( [ min(this.psd)/10 , max(ylim()) ] )
        xlabel( r'$f$ (Hz)' )
        ylabel( r'$\sqrt{S_n(f)}$  and  $2|\tilde{h}(f)|\sqrt{f}$' )
        legend( frameon=False )


    # Precalculate items related to the signals
    def __set_waveform_fields__(this):

        # NOTE that some fields are set in the __parse_inputs__ method

        # Import useful things
        from numpy import sqrt

        # Calculate the combination of SIGNAL + and x as determined by the signal_polarization
        this.signal['response'] = this.calc_detector_response( this.signal['polarization'], this.signal['+'], this.signal['x'] )

        # Calculate the combination of TEMPLATE + and x as determined by the signal['polarization']
        this.template['response'] = this.calc_detector_response( this.template['polarization'], this.template['+'], this.template['x'] )

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
    def calc_template_pol_optimized_match( this, **kwargs ):

        # Import useful things
        from numpy import sqrt,dot,real,log,diff
        from numpy.fft import ifft

        # Handle signal polarization input; use constructor value as default
        this.apply( **kwargs )

        #
        this.template['+norm'] = this.calc_norm( this.template['+'] )
        this.template['xnorm'] = this.calc_norm( this.template['x'] )
        #
        if (this.template['+norm']==0) or (this.template['xnorm']==0) :
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
        # from matplotlib import pyplot as pp
        # pp.plot( numerator/ (denominator*2.0) )
        # print sqrt( numerator.max() / (denominator*2.0) )
        # print this.signal['norm']
        template_pol_optimized_match = sqrt( numerator.max() / (denominator*2.0) ) / (this.signal['norm'] if this.signal['norm'] != 0 else 1)

        #
        ans = template_pol_optimized_match
        return ans

    # Claculate optimal snr
    def calc_optsnr(this,a):
        #
        from numpy import sqrt
        # See Maggiori, p. 345 -- a factor of 2 comes from the definition of the psd,
        # and another from the double sidedness of the spectrum
        optsnr = 4 * sqrt( this.calc_overlap(a,method='trapz') )
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
    def calc_detector_response(this,psi,h_plus,h_cross):
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
        this.psd = this.__psd_fun__(this.f)


    # Evlaute a desired psd
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
        psd_names_for_tabulated_data = ['aligo','150914']
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
                elif '091' in psd_name:
                    data_path = nrutils.__path__[0]+'/data/H1L1-AVERAGE_PSD-1127271617-1027800.txt'
                #
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
                psd_fun = lambda f: sqrt( iligo(f,version=2) )


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


    #
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


    # Unpack and Validate inputs
    def __parse_inputs__(this, template_wfarr, signal_wfarr, fmin, fmax, signal_polarization, template_polarization, psd_name, verbose):

        # Import useful things
        from numpy import allclose,diff

        # Store the verbose input to the current object
        this.verbose = verbose

        #
        if this.verbose:
            alert('Verbose mode on.','match')

        #
        this.fmin = fmin if fmin is not None else this.fmin
        this.fmax = fmax if fmax is not None else this.fmax

        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        # Check for equal array domain     #
        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        if (signal_wfarr is not None) and (template_wfarr is not None):
            signal_f,template_f = signal_wfarr[:,0],template_wfarr[:,0]
            signal_df,template_df = diff(signal_f[:2])[0],diff(template_f[:2])[0]
            if len(signal_f) != len(template_f):
                print len(signal_f),len(template_f)
                error( 'Frequency columns of waveform arrays are not equal in length. You may wish to interpolate to ensure a common frequency domain space.' )
            if not allclose(signal_f,template_f):
                error("Values in the tempate frequncy column are not all close to values in the signal frequency column. This should not be the case.")

            # Crop arrays between fmin and fmax, and then store to the current object
            f = signal_f
            mask = (f>=fmin) & (f<=fmax)
            this.f = f[mask].real # NOTE The signal_wfarr is complex typed; for the frequency column the imag parts are zero; here we explicitely cast frequencies as real to avoid numpy type errors when evaluating the psd

            # Group plus and cross into dictionaries
            this.signal,this.template = {},{}
            this.signal['+'] = signal_wfarr[mask,1]
            this.signal['x'] = signal_wfarr[mask,2]
            this.template['+'] = template_wfarr[mask,1]
            this.template['x'] = template_wfarr[mask,2]

        # Store the psd name input
        this.psd_name = psd_name

        # Group pluss and cross into dictionaries
        this.signal['polarization'] = signal_polarization if signal_polarization is not None else 0
        this.template['polarization'] = template_polarization if template_polarization is not None else 0

        '''
        The core functions of this class will operate on this.f, this.signal and this.template
        '''

        #
        return None
