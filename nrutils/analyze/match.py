# Import useful things
from nurtils.core.basics import *

#
class match:

    #
    def __init__( this,                       # The current object
                  template_wfarr = None,      # column format [ f + x ]
                  signal_wfarr = None,        # column format [ f + x ]
                  psd_name = None,            # determines which psd to use; default is advLigo
                  fmin = None,
                  fmax = None,
                  signal_polarization = None,
                  input_units = None,
                  verbose = True ):

        # Unpack and Validate inputs
        this.__parse_inputs__(template_wfarr,signal_wfarr,fmin,fmax,signal_polarization,verbose)

        # Calculate PSD at desired frequencies; store to current object
        this.__set_psd__(psd_name)

        # Calculate and store relevant information about the tempalte and signal waveforms
        this.__set_waveform_fields__()

    # Precalculate items related to the signals
    def __set_waveform_fields__(this):

        # Import useful things
        from numpy import sqrt
        # Calculate the combination of SIGNAL + and x as determined by the signal_polarization
        this.signal['response'] = this.calc_detector_response( this.signal_polarization, this.signal['+'], this.signal['x'] )
        # Calculate the combination of TEMPLATE + and x as determined by the signal_polarization
        this.template['response'] = this.calc_detector_response( this.signal_polarization, this.template['+'], this.template['x'] )
        # Calculate the signal's optimal snr
        this.signal['norm'] = this.calc_norm( this.signal['response'] )
        # Store Optimal SNRs
        this.signal['optimal_snr'] = this.signal['norm']
        this.template['optimal_snr'] = this.calc_norm( this.template['response'] )
        # Calculate the normed signal with respect to the overlap
        this.signal['normalized_response'] = this.signal['response'] / this.signal['norm']
        # Calculate the normed template with respect to the overlap
        this.template['normalized_response'] = this.template['response'] / this.template['norm']

        return None

    # Calculate overlap optimized over template polarization
    def calc_template_pol_optimized_match(this):

        # Import useful things
        from numpy import sqrt,dot,real,log
        from numpy.fft import ifft

        #
        this.template['+norm'] = this.calc_norm( this.template['+'] )
        this.template['xnorm'] = this.calc_norm( this.template['x'] )
        #
        if (this.template['+norm']==0) or (this.template['xnorm']==0) :
            error('Neither + nor x of template can be zero for all frequencies.')
        #
        normalized_template_plus  = this.template['+']/this.template['+norm']
        normalized_template_cross = this.template['x']/this.template['xnorm']
        IPC = real( this.overlap(normalized_template_plus,normalized_template_cross) )

        # Padding for effective sinc interpolation of ifft
        fftlen = 2 ** ( int(log(dataL)/log(2)) + 1.0 + 3.0 )
        #
        integrand = lambda a: a.conj() * this.signal['normalized_response'] / this.psd
        rho_p = fftlen * ifft( integrand(normalized_template_plus ) , n=fftlen )
        rho_c = fftlen * ifft( integrand(normalized_template_cross) , n=fftlen )
        #
        gamma = ( rho_p * rho_c.conj() ).real
        rho_p2 = abs( rho_p ) ** 2
        rho_c2 = abs( rho_c ) ** 2
        sqrt_part = sqrt( (rho_p2-rho_c2)**2 + 4*(IPC*rho_p2-gamma)*(IPC*rho_c2-gamma) )
        numerator = rho_p2 - 2.0*IPC*gamma + rho_c2 + sqrt_part
        denominator = 1.0 - IPC**2
        template_pol_optimized_match = sqrt( numerator.max() / (denominator*2.0) ) / this.signal(norm)

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
        if (this.psd.shape == b.shape) or (this.psd.shape == a.shape):
            error('vector shapes not compatible with this objects psd shape')
        #
        integrand = ( abs(a)**2 if b is None else a.conj()*b ) / this.psd
        #
        method = 'trapz' if method is None or not isinstance(method,str) else method
        if not method in ('trapz','spline'): error('unknown integration method input; must be spline or trapz')
        #
        if method in ('trapz'):
            overlap = trapz(integrand,this.f)
        elif method in ('spline'):
            overlap = spline(this.f,integrand).integral(this.fmin,this.fmax)
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
    def __set_psd__(this,psd_name):
        this.__set_psd_fun__(psd_name)
        this.psd = this.__psd_fun__(this.f)

    # Evlaute a desired psd
    def __set_psd_fun__(this,psd_name):
        '''
        In an if-else sense, the psd_name will be interpreterd as:
        * a short string relating to know psd values
        * the path location of an ascii file containing psd values
        * an array of frequency and psd values
        '''
        # Import useful things
        from numpy import ndarray
        import nrutils
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        #
        known_psd_names = ['aligo','150914']
        not_valid_msg = 'psd_name must either be short string name in the set %s, the path location of psd data, or a numpy array containing with columns [f_phys_hz psd_values]. Something else was found.'%known_psd_names
        # If psd_name is a string
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
        if psd_arry.shape[-1] is 2:
            psd_f,psd_vals = psd_array[:,0],psd_array[:,1]
        else:
            error('Improperly formatted psd array given. Instead of having two columns, it has %i'%psd_arry.shape[-1])

        # Create an interpolation of the PSD data
        # NOTE that this function is stored to the current object
        this.__psd_fun__ = spline(psd_f,psd_vals)

        #
        return None

    # Unpack and Validate inputs
    def __parse_inputs__(this,template_arr,signal_arr,fmin,fmax,signal_polarization,verbose):

        # Import useful things
        from numpy import allclose

        # Store the verbose input to the current object
        this.verbose = verbose

        #
        this.signal_polarization = signal_polarization
        this.fmin = fmin
        this.fmax = fmax

        #
        if this.verbose:
            alert('Verbose mode on.','match')

        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        # Check for equal array domain     #
        #-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-%%-#
        signal_f,temaplte_f = signal_arr[:,0],template_arr[:,0]
        signal_df,template_df = diff(signal_f[:2])[0],diff(template_f[:2])[0]
        if len(signal_f) is not len(template_f):
            error( 'Frequency columns of waveform arrays are not equal in length. You may wish to interpolate to ensure a common frequency domain space.' )
        if not allclose(signal_f,temaplte_f):
            error("Values in the tempate frequncy column are not all close to values in the signal frequency column. This should not be the case.")

        # Crop arrays between fmin and fmax, and then store to the current object
        f = signal_f
        mask = (f>=fmin) & (f<=fmax)
        this.f = f[mask]
        # Group pluss and cross into dictionaries
        this.signal,this.template = {},{}
        this.signal['+'] = signal_arr[mask,1]
        this.signal['x'] = signal_arr[mask,2]
        this.template['+'] = template_arr[mask,1]
        this.template['x'] = template_arr[mask,2]

        '''
        The core functions of this class will operate on this.f, this.signal and this.template
        '''

        #
        return None
