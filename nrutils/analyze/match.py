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

        # Unpack and Validate inputs
        this.__parse_inputs__(template_wfarr,signal_wfarr,fmin,fmax,signal_polarization,template_polarization, psd_name,verbose)

        # Calculate PSD at desired frequencies; store to current object
        this.__set_psd__()

        # Calculate and store relevant information about the tempalte and signal waveforms
        this.__set_waveform_fields__()

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
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot,show,xlabel,ylabel,title,\
                                      legend,xscale,yscale,xlim,ylim,figure
        # Setup Figure
        figure( figsize=1.8*array([4,4]) )
        # Plot
        plot( this.f, this.psd, '-k', label='PSD' )
        plot( this.f, 2*sqrt(this.f)*abs(this.signal['response']), label='signal' )
        #
        xscale('log'); yscale('log')
        xlim(lim(this.f))
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
        # Store Optimal SNRs
        this.signal['optimal_snr'] = this.calc_norm( this.signal['response'],method='trapz' )
        this.template['optimal_snr'] = this.calc_norm( this.template['response'],method='trapz' )
        # Calculate the normed signal with respect to the overlap
        this.signal['normalized_response'] = this.signal['response'] / this.signal['norm']
        # Calculate the normed template with respect to the overlap
        this.template['normalized_response'] = this.template['response'] / this.template['norm']

        return None

    # Calculate overlap optimized over template polarization
    def calc_template_pol_optimized_match(this,signal_polarization=None):

        # Import useful things
        from numpy import sqrt,dot,real,log,diff
        from numpy.fft import ifft

        # Handle signal polarization input; use constructor value as default
        signal_polarization = this.signal['polarization'] if signal_polarization is None else signal_polarization
        this.__set_waveform_fields__()

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
        template_pol_optimized_match = sqrt( numerator.max() / (denominator*2.0) ) / this.signal['norm']

        #
        ans = template_pol_optimized_match
        return ans

    # Calculate noise curve weighted norm (aka Optimal SNR)
    def calc_norm(this,a,method=None):
        #
        from numpy import sqrt
        #
        norm = sqrt( this.calc_overlap(a,a,method=method) )
        #
        ans = norm
        return ans

    # Calculate the PSD weighted inner-product between to waveforms
    def calc_overlap(this,a,b=None,method=None):
        #
        from numpy import trapz
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        #
        b = a if b is None else b
        #
        if (this.psd.shape[0] != b.shape[0]) or (this.psd.shape[0] != a.shape[0]):
            error('vector shapes not compatible with this objects psd shape')
        #
        integrand = ( abs(a)**2 if b is a else a.conj()*b ) / this.psd
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
        from numpy import ndarray,loadtxt
        import nrutils
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        #
        known_psd_names = ['aligo','150914']
        not_valid_msg = 'psd_name must either be short string name in the set %s, the path location of psd data, or a numpy array containing with columns [f_phys_hz psd_values]. Something else was found.'%known_psd_names
        # If psd_name is a string
        psd_name = this.psd_name
        if isinstance(psd_name,str):

            # If psd_name is a standard psd name
            if psd_name.lower() in known_psd_names:
                # Load the corresponding data
                psd_name = psd_name.lower()
                if 'al' in psd_name:
                    data_path = nrutils.__path__[0]+'/data/ZERO_DET_high_P.dat'
                elif '091' in psd_name:
                    data_path = nrutils.__path__[0]+'/data/H1L1-AVERAGE_PSD-1127271617-1027800.txt'
                #
                psd_arr = loadtxt( data_path )
            elif not isinstance(psd_name,ndarray):
                error(not_valid_msg)

        elif isinstance(psd_name,ndarray):
            # Else if it's an array of psd data
            psd_arr = psd_name
        else:
            error(not_valid_msg)

        # Validate and unpack the psd array
        if psd_arr.shape[-1] is 2:
            psd_f,psd_vals = psd_arr[:,0],psd_arr[:,1]
            # from matplotlib import pyplot as pp
            # pp.plot( psd_f, psd_vals, label='aLIGO:ZERO_DET_high_P' )
            # pp.yscale('log')
            # pp.xscale('log')
            # pp.legend(frameon=False)
            # pp.show()
        else:
            error('Improperly formatted psd array given. Instead of having two columns, it has %i'%psd_arry.shape[-1])

        # Create an interpolation of the PSD data
        # NOTE that this function is stored to the current object
        this.__psd_fun__ = spline(psd_f,psd_vals)

    # Unpack and Validate inputs
    def __parse_inputs__(this, template_arr, signal_arr, fmin, fmax, signal_polarization, template_polarization, psd_name, verbose):

        # Import useful things
        from numpy import allclose,diff

        # Store the verbose input to the current object
        this.verbose = verbose

        #
        if this.verbose:
            alert('Verbose mode on.','match')

        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        # Check for equal array domain     #
        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        signal_f,template_f = signal_arr[:,0],template_arr[:,0]
        signal_df,template_df = diff(signal_f[:2])[0],diff(template_f[:2])[0]
        if len(signal_f) != len(template_f):
            print len(signal_f),len(template_f)
            error( 'Frequency columns of waveform arrays are not equal in length. You may wish to interpolate to ensure a common frequency domain space.' )
        if not allclose(signal_f,template_f):
            error("Values in the tempate frequncy column are not all close to values in the signal frequency column. This should not be the case.")

        # Crop arrays between fmin and fmax, and then store to the current object
        f = signal_f
        mask = (f>=fmin) & (f<=fmax)
        this.f = f[mask].real # NOTE The signal_arr is complex typed; for the frequency column the imag parts are zero; here we explicitely cast frequencies as real to avoid numpy type errors when evaluating the psd

        # Store the psd name input
        this.psd_name = psd_name

        # Group pluss and cross into dictionaries
        this.signal,this.template = {},{}
        this.signal['+'] = signal_arr[mask,1]
        this.signal['x'] = signal_arr[mask,2]
        this.template['+'] = template_arr[mask,1]
        this.template['x'] = template_arr[mask,2]
        this.signal['polarization'] = signal_polarization
        this.template['polarization'] = template_polarization

        '''
        The core functions of this class will operate on this.f, this.signal and this.template
        '''

        #
        return None
