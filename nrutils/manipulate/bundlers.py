from __future__ import print_function

'''
module for routines that bundle many wavforms into a single obejct for output
'''


def nr2h5( nr_strain_data, nr_meta_data, output_path=None, verbose=False ):
    '''

    Example formatting for inputs:

    nr_strain_data = { ... (2,2):{'amp':strain_amplitude_array,'phase':strain_phase_array,'t':time_series}, (2,1): ... }

    nr_meta_data   = { 'NR-group':'Your Group Name', 'mass1':0.5, ... }

    NOTE: the time series must all be the same

    '''
    this_module = 'nr2h5'

    # import useful things
    from os import system, remove, makedirs, path
    import shutil
    from os.path import dirname, basename, isdir, realpath
    from numpy import array,ones,pi,loadtxt,hstack
    from numpy.linalg import norm
    from os.path import expanduser
    from numpy import array, diff
    import romspline
    import h5py

    #
    def alert(msg,base):
        if verbose:
            print( "*(%s)>> %s" % (base,msg))


    # -------- Define useful things for unit scaling -------- #
    # Nme intermediate function for unit scaling
    G = 6.67428e-11; c = 2.99792458e8; mass_sun = 1.98892e30;
    def mass_sec( M ): return M*G*mass_sun/(c*c*c)
    # Convert code frequency to physical frequency
    def physf( f, M ): return f/mass_sec(M)
    # ------------------------------------------------------- #


    # ---- Define useful things for input checking & file handling ---- #

    # Function to convert string to bytes
    def to_bytes(string):
        return bytes(string,encoding="utf-8")

    # Make "mkdir" function for directories
    def mkdir(dir):
        # Check for directory existence; make if needed.
        if not path.exists(dir):
            makedirs(dir)
        # Return status
        return path.exists(dir)

    #
    def parent(path):
        '''
        Simple wrapper for getting absolute parent directory
        '''
        import os
        return os.path.abspath(os.path.join(path, os.pardir))+'/'


    # make sure that both inputs are dictionaries
    if ( not isinstance(nr_strain_data,dict) ) or ( not isinstance(nr_strain_data,dict) ):
        msg = 'both inputs must be dictionaries with the apporpirate fields '
        raise ValueError( msg )

    # make sure that all of the time series are the same length and have the same values
    alert('Validating inputs.',this_module )
    required_strain_keys = ['amp','phase','t']; t_length_record = []
    for a in nr_strain_data:
        for b in nr_strain_data[a]:
            if not (b in required_strain_keys):
                msg = 'all multipole keys must contain "amp" and "phase" keys, and the "t" key for time'
                raise ValueError(msg)
            if 't' in nr_strain_data[a]:
                t_length_record.append( len(nr_strain_data[a]['t']) )

    # Verify that all time arrays have the same length
    if sum( diff( array( t_length_record ) ) ) != 0:
        msg = 'not all time arrays are of the same length in the nr_strain_data input'
        raise ValueError(msg)
    # NOTE: More erro checking possible.

    # Valudate output_path
    if not ('.h5' == output_path[-3:]):
        msg = 'output_path input MUST be the full path of the h5 file you wish to create'
        raise ValueError(msg)

    # List required metadata keys
    required_metadata_keys = [ 'NR-group','type','name',
                      'object1','object2',
                      'mass1','mass2','eta',
                      'spin1x','spin1y','spin1z',
                      'spin2x','spin2y','spin2z',
                      'LNhatx','LNhaty','LNhatz',
                      'nhatx','nhaty','nhatz',
                      'f_lower_at_1MSUN','eccentricity','PN_approximant' ]

    # Check for correct metadata keys
    for key in required_metadata_keys:
        if not (key in nr_meta_data):
            #
            msg = '"%s" not present in fist input, but must be.' % key
            raise ValueError( msg )
    # ----------------------------------------------------------------- #



    # ----------------------------------------------------------------- #
    # Output directory of hdf5 file
    if output_path is None:
        #output_path = '/Users/book/JOKI/Libs/KOALA/nrutils_dev/review/data/'
        output_path = './new_h5_file.h5'

    # --> Make sure the output path exists
    mkdir(parent(output_path))

    # Define a temporary directory for amplitude and phase rom files
    tmp_dir = './.nr2h5tmp/'

    # Make the temp dir if it does not already exist
    alert('Making temporary directory at "%s".'%(tmp_dir),this_module )
    mkdir(tmp_dir)


    # Initiate h5 file for seeding attributes and groups
    alert('Initiating h5 file at "%s".'%(output_path),this_module )
    h5f = h5py.File( output_path, 'w' )

    # Define attributes
    alert('Seeding metadata from input:',this_module )
    for key in nr_meta_data:
        if verbose: print( "\t\t %s = %s" % (key,nr_meta_data[ key ]))
        h5f.attrs[ key ] = nr_meta_data[ key ]

    # Define functoin to create spline and store temporary h5 file
    def write_spline( output_location, domain_data, feature_data ):
        # Create romspline object of multipole data
        spline = romspline.ReducedOrderSpline( domain_data, feature_data, verbose=False )
        spline.write( output_location )

    # Amplitude and phase roms must be created, stored to temporary files, and then added to the main hdf5 file above. This is a pain.
    if (2,2) in nr_strain_data:
        time_data = nr_strain_data[(2,2)]['t']
    else:
        time_data = nr_strain_data[ lm[0] ]['t']
    alert('Storing NR multipoles to h5 file. Tempoary h5 files will be created and deleted on the fly.',this_module )
    for lm in nr_strain_data:

        if verbose: print( ' ... (l,m)=(%s,%s)' % lm)

        # Unpack multipole indeces
        l = lm[0]; m = lm[1]

        # Create romspline object of multipole amplitude data
        amp_tmp = tmp_dir + 'amp_l%im%i.h5' % lm
        write_spline( amp_tmp, time_data, nr_strain_data[(l,m)]['amp'] )

        # Create romspline object of multipole phase data
        phase_tmp = tmp_dir + 'phase_l%im%i.h5' % lm
        write_spline( phase_tmp, time_data, nr_strain_data[(l,m)]['phase'] )

        # Open the temporary h5 file, and then copy its high level group contents to the main h5 file under the appropriate group
        tmp_amp_h5 = h5py.File( amp_tmp , 'r')
        h5py.h5o.copy( tmp_amp_h5.id , to_bytes('/'), h5f.id, to_bytes('/amp_l%i_m%i' % lm) )
        #
        tmp_phase_h5 = h5py.File( phase_tmp , 'r')
        h5py.h5o.copy( tmp_phase_h5.id , to_bytes('/'), h5f.id, to_bytes('/phase_l%i_m%i' % lm) )

        # delete temporary h5 files
        remove( amp_tmp ); remove( phase_tmp )

    # Remove the temporary directory
    alert('Removing temporary directory at "%s".'%(tmp_dir),this_module )
    shutil.rmtree(tmp_dir)

    #
    alert('I have created an hdf5 file at "%s"'%(output_path),this_module )
