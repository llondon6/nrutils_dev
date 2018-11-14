
'''
Modules for Numerical Relativity Simulation Catalog:

* catalog: builds catalog given a cinfiguration file, or directory containing many configuration files.

* scentry: class for simulation catalog entry (should include io)

'''

#
from nrutils.core import global_settings
from nrutils.core.basics import *
from nrutils.core import M_RELATIVE_SIGN_CONVENTION
import warnings,sys

#
class __gconfig__:

    '''
    This class is to play the role of config files by:
    1. Loading and storing static fields within each config file
    2. Allowing the user to change fields such as catalog directories
    3. Classes and methods within nrsc.py will use a member of this class to navigate the catalog database files.
    4. Upon import, nrutils will load the user settings, with the user's previoius settings state carried over.
    '''

    #
    def __init__( this, verbose=False ):

        # Import usefuls
        from glob import glob as find
        from os.path import expanduser,join,basename,isdir,isfile
        from shutil import copyfile as cp
        import dill, pickle

        # Load all default config file locations
        config_paths = find( global_settings.config_path+'*.ini' )

        # Let the people know
        if verbose: alert('Found %i config (ini) files in "%s"'%(len(config_paths),cyan(global_settings.config_path)),'gconfig')

        # Create a folder to store user settings files
        user_settings_path = expanduser('~/.nrutils_user_settings/')
        is_new_settings_filder = not isdir(user_settings_path)
        mkdir(user_settings_path,verbose=verbose)

        # Copy each default config file to the user's local config_path
        # * existing files will not be overwritten
        user_config_files = []
        for config_path in config_paths:
            config_file_location = join(user_settings_path,basename(config_path))
            user_config_files.append(config_file_location)
            if not isfile(config_file_location):
                if verbose: print '>> copying "%s" to "%s" to make up for missing config'%(blue(config_path),blue(config_file_location))
                cp( config_path, config_file_location )

        # Create a smart object representation of each config
        soconf = []
        for ucf in user_config_files:
            #
            so = scconfig( ucf )
            #
            soconf.append( so )

        # Define fucntion that return the first existing dir in a list of dirs
        def take_existing( flist ):
            if not isinstance(flist,(list,tuple)):
                flist = [flist]
            for f in flist:
                if isdir(f):
                    return f

        # Determine the catalog directories for each, and store each catalog dir
        for so in soconf:
            #
            so.catalog_dir = take_existing( so.catalog_dir )

        # Store the list of config objects
        this.configs = soconf

        # Define the location of the user's living config bundle
        user_config_obj_path = join( user_settings_path, 'global_config' )
        # store the current object to this location
        with open( user_config_obj_path, 'wb' ) as f:
            pickle.dump( this , f )


# Class representation of configuration files. The contents of these files define where the metadata for each simulation is stored, and where the related NR data is stored.
class scconfig(smart_object):

    # Create scconfig object from configuration file location
    def __init__(this,config_file_location=None,overwrite=True,verbose=True):

        # Required fields from smart_object
        this.source_file_path = []
        this.source_dir = []
        this.overwrite = overwrite
        this.verbose = verbose

        # call wrapper for constructor
        this.config_file_location = config_file_location
        this.reconfig()

        # Possibly useful toggle for labeling existance of catalog_dir
        this.is_valid = True

    # The actual constructor: this will be called within utility functions so that scentry objects are configured with local settings (i.e. global_settings).
    def reconfig(this):

        #
        from os.path import expanduser

        #
        if this.config_file_location is None:
            msg = '(!!) scconfig objects cannot be initialted/reconfigured without a defined "config_file_location" location property (i.e. string where the related config file lives); you may be recieving this message if youve created an instance outside of the core routines'
            raise ValueError(msg)

        #
        stale_config_exists = os.path.exists( this.config_file_location )
        if not stale_config_exists:
            # Try to refresh the config location using the user's current settings (i.e. global_settings) (see __init__.py in nrutils/core)
            config_path = global_settings.config_path
            stale_config_name = this.config_file_location.split('/')[-1]
            fresh_config_file_location = expanduser( config_path + '/' + stale_config_name )
            # If the new config path exists, then store it and use it to reconfigure the current object
            if os.path.exists( fresh_config_file_location ):
                this.config_file_location = fresh_config_file_location
            else:
                error(  'nrutils noticed that the config objects config_location of "%s" does not extist. It then tried to determine the correct location from your paths.ini file (see in package nrutils/setting/paths.ini). This estimate location was determined to be "%s". Sadly, this file was also not found by the OS. "%s"' % ( magenta(this.config_file_location), magenta(fresh_config_file_location) )  )

        # learn the contents of the configuration file
        if os.path.exists( this.config_file_location ):
            this.learn_file( this.config_file_location, comment=[';','#'] )
            # validate the information learned from the configuration file against minimal standards
            this.validate()
            this.config_exists = True
        else:
            msg = 'There is a simulation catalog entry (scentry) object which references \"%s\", however such a file cannot be found by the OS. The related scentry object will be marked as invalid.'%cyan(this.config_file_location)
            this.config_exists = False
            warning(msg,'scconfig.reconfig')

        # Select existing catalof dir
        if isinstance(this.catalog_dir,list):
            found_directory = False
            for d in this.catalog_dir:
                if os.path.isdir(d):
                    this.catalog_dir = d
                    found_directory = True
                    break
            if not found_directory:
                warning('You system does not have access to the following catalog directories as defined in settings:\n%s\n If you\'d like access to this catalog, please update your settings or manually mount the desired location.'%(cyan(str(this.catalog_dir))))


        # In some cases, it is useful to have this function return this
        return this

    # Get location of handler file based on handler name in config ini
    def get_handler_location(this):
        '''Get location of handler file based on handler name in config ini'''
        handler_location = global_settings.handler_path + this.handler_name + '.py'
        # alert('The handler location is %s'%yellow(handler_location),header=True)
        return handler_location

    # Validate the config file against a minimal set of required fields.
    def validate(this):

        # Import useful things
        from os.path import expanduser

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        # each scconfig object (and the realted file) MUST have the following attributes
        required_attrs = [ 'institute',                 # school or collaboration authoring run
                           'metadata_id',               # unique string that defines metadata files
                           'catalog_dir',               # local directory where all simulation folders are stored
                                                        # this directory allows catalog files to be portable
                           'data_file_name_format',     # formatting string for referencing l m and extraction parameter
                           'handler_name',              # name of handler file WITHOUT extension in global_settings.handler_path
                                                        # learn_metadata functions
                           'is_extrapolated',           # users should set this to true if waveform is extrapolated
                                                        # to infinity
                           'is_rscaled',                # Boolean for whether waveform data are scaled by extraction radius (ie rPsi4)
                           'default_par_list' ]   # list of default parameters for loading: default_extraction_parameter, default_level. NOTE that list must be of length 2

        # Make sure that each required attribute is a member of this objects dictionary representation. If it's not, throw an error.
        for attr in required_attrs:
            if not ( attr in this.__dict__ ):
                msg = '(!!) Error -- config file at %s does NOT contain required field %s' % (  magenta(this.config_file_location), attr )
                raise ValueError(msg)

        # Make sure that data_file_name_format is list of strings. The intention is to give the user the ability to define multiple formats for loading. For example, the GT dataset may have files that begin with Ylm_Weyl... and others that begin with mp_Weylscalar... .
        if isinstance( this.data_file_name_format, str ):
            this.data_file_name_format = [this.data_file_name_format]
        elif isinstance(this.data_file_name_format,list):
            for k in this.data_file_name_format:
                if not isinstance(k,str):
                    msg = '(!!) Error in %s: each element of data_file_name_format must be character not numeric. Found data_file_name_format = %s' % (magenta(this.config_file_location),k)
                    raise ValueError(msg)
                if False: # NOTE that this is turned off becuase it is likely not the appropriate way to check. More thought needed. Original line: len( k.split('%i') ) != 4:
                    msg = '(!!) Error in %s: All elements of data_file_name_format must have three integer formatting tags (%%i). The offending entry is %s.' % ( magenta(this.config_file_location), red(k) )
                    raise ValueError(msg)
        else:
            msg = '(!!)  Error in %s: data_file_name_format must be comma separated list.' %  magenta(this.config_file_location)

        # Make sure that catalog_dir is string
        if not isinstance( this.catalog_dir, (list,str) ):
            msg = 'catalog_dir values must be string or list'
            error(red(msg),thisfun)

        # If catalog_dir is list of dirs, then select the first one that exists
        if isinstance( this.catalog_dir, list ):
            # if this.verbose: warning('Multiple catalog directories found. We will scan through the related list, and then store first the catalog_dir that the OS can find.')
            from os.path import isdir,expanduser
            for d in this.catalog_dir:
                d = expanduser(d)
                if isdir(d):
                    this.catalog_dir = d
                    # if this.verbose: warning('Selecting "%s"'%cyan(d))
                    break


        if 2 != len(this.default_par_list):
            msg = '(!!) Error in %s: default_par_list must be list containing default extraction parameter (Numeric value) and default level (also Numeric in value). Invalide case found: %s' % (magenta(this.config_file_location),list(this.default_par_list))
            raise ValueError(msg)

        # Make sure that all directories end with a forward slash
        for attr in this.__dict__:
            if 'dir' in attr:
                if isinstance(this.__dict__[attr],str):
                    if this.__dict__[attr][-1] != '/':
                        this.__dict__[attr] += '/'
                elif isinstance(this.__dict__[attr],(list,tuple)):
                    for k in range(len(this.__dict__[attr])):
                        if this.__dict__[attr][k] != '/':
                            this.__dict__[attr][k] += '/'
        # Make sure that user symbols (~) are expanded
        for attr in this.__dict__:
            if ('dir' in attr) or ('location' in attr):
                if isinstance(this.__dict__[attr],str):
                    this.__dict__[attr] = expanduser( this.__dict__[attr] )
                elif isinstance(this.__dict__[attr],list):
                    for k in this.__dict__[attr]:
                        if isinstance(k,str):
                            k = expanduser(k)

# Class for simulation catalog e.
class scentry:

    # Create scentry object given location of metadata file
    def __init__( this, config_obj, metadata_file_location, static=False, verbose=True ):

        # Keep an internal log for each scentry created
        this.log = '[Log for %s] The file is "%s".' % (this,metadata_file_location)

        # Store primary inputs as object attributes
        this.config = config_obj
        this.metadata_file_location = metadata_file_location

        # Toggle for verbose mode
        this.verbose = verbose

        # Validate the location of the metadata file: does it contain waveform information? is the file empty? etc
        this.isvalid = this.validate() if config_obj else False

        # Toggle for whether configuration object is dynamic (dynamically reference ini file) or static ( keep values stored in object at all times -- see reconfig() )
        this.static = static

        # If valid, learn metadata. Note that metadata property are defined as none otherise. Also NOTE that the standard metadata is stored directly to this object's attributes.
        this.raw_metadata = None
        if this.isvalid is True:
            #
            if this.verbose: print '## Working: %s' % cyan(metadata_file_location)
            this.log += ' This entry\'s metadata file is valid.'

            # # i.e. learn the meta_data_file
            # this.learn_metadata(); # raise(TypeError,'This line should only be uncommented when debugging.')
            # this.label = sclabel( this )

            try:
                this.learn_metadata()
                this.label = sclabel( this )
            except:
                emsg = sys.exc_info()[1].message
                this.log += '%80s'%' [FATALERROR] The metadata failed to be read. There may be an external formatting inconsistency. It is being marked as invalid with None. The system says: %s'%emsg
                if this.verbose: warning( 'The following error message will be logged: '+red(emsg),'scentry')
                this.isvalid = None # An external program may use this to do something
                this.label = 'invalid!'

        elif this.isvalid is False:
            if config_obj:
                if this.verbose: print '## The following is '+red('invalid')+': %s' % cyan(metadata_file_location)
                this.log += ' This entry\'s metadta file is invalid.'

    # Method to load handler module
    def loadhandler(this):
        # Import the module
        from imp import load_source
        handler_module = load_source( '', this.config.get_handler_location() )
        # Validate the handler module: it has to have a few requried methods
        required_methods = [ 'learn_metadata', 'validate', 'extraction_map' ]
        for m in required_methods:
            if not ( m in handler_module.__dict__ ):
                msg = 'Handler module must contain a method of the name %s, but no such method was found'%(cyan(m))
                error(msg,'scentry.validate')
        # Return the module
        return handler_module

    # Validate the metadata file using the handler's validation function
    def validate(this):

        # import validation function given in config file
        # Name the function representation that will be used to load the metadata file, and convert it to raw and standardized metadata
        validator = this.loadhandler().validate

        # vet the directory where the metadata file lives for: waveform and additional metadata
        status = validator( this.metadata_file_location, config = this.config, verbose=this.verbose )

        #
        return status

    # Standardize metadata
    def learn_metadata(this):

        #
        from numpy import allclose

        # Load the handler for this entry. It will be used multiple times below.
        handler = this.loadhandler()

        # Name the function representation that will be used to load the metadata file, and convert it to raw and standardized metadata
        learn_institute_metadata = handler.learn_metadata

        # Eval and store standard metadata
        [standard_metadata, this.raw_metadata] = learn_institute_metadata( this.metadata_file_location )

        # Validate the standard metadata
        required_attrs = [ 'date_number',   # creation date (number!) of metadata file
                           'note',          # informational note relating to metadata
                           'madm',          # initial ADM mass = m1+m2 - initial binding energy
                           'b',             # initial orbital separation (scalar: M)
                           'R1', 'R2',      # initial component masses (scalars: M = m1+m2)
                           'm1', 'm2',      # initial component masses (scalars: M = m1+m2)
                           'P1', 'P2',      # initial component linear momenta (Vectors ~ M )
                           'L1', 'L2',      # initial component angular momental (Vectors ~ M)
                           'S1', 'S2',      # initial component spins (Vectors ~ M*M)
                           'mf', 'Sf',      # Final mass (~M) and final dimensionful spin (~M*M)
                           'Xf', 'xf' ]     # Final dimensionless spin: Vector,Xf, and *Magnitude*: xf = sign(Sf_z)*|Sf|/(mf*mf) (NOTE the definition)

        for attr in required_attrs:
            if attr not in standard_metadata.__dict__:
                msg = '(!!) Error -- Output of %s does NOT contain required field %s' % ( this.config.get_handler_location(), attr )
                raise ValueError(msg)

        # Add useful fields: chi1 chi2 and eta
        standard_metadata.X1 = standard_metadata.S1 / (standard_metadata.m1**2)
        standard_metadata.X2 = standard_metadata.S2 / (standard_metadata.m2**2)
        standard_metadata.eta = standard_metadata.m1*standard_metadata.m2 / ( standard_metadata.m1+standard_metadata.m2 )

        # Confer the required attributes to this object for ease of referencing
        for attr in standard_metadata.__dict__.keys():
            setattr( this, attr, standard_metadata.__dict__[attr] )

        # tag this entry with its inferred setname
        this.setname = this.raw_metadata.source_dir[-1].split( this.config.catalog_dir )[-1].split('/')[0]

        # tag this entry with its inferred simname
        this.simname = this.raw_metadata.source_dir[-1].split('/')[-1] if this.raw_metadata.source_dir[-1][-1]!='/' else this.raw_metadata.source_dir[-1].split('/')[-2]

        # tag this entry with the directory location of the metadata file. NOTE that the waveform data must be reference relative to this directory via config.data_file_name_format
        this.relative_simdir = this.raw_metadata.source_dir[-1].split( this.config.catalog_dir )[-1]

        # NOTE that is is here that we may infer the default extraction parameter and related extraction radius

        # Load default values for extraction_parameter and level (e.g. resolution level)
        # NOTE that the special method defined below must take in an scentry object, and output extraction_parameter and level
        special_method = 'infer_default_level_and_extraction_parameter'
        if special_method in handler.__dict__:
            # Let the people know
            if this.verbose:
                msg = 'The handler is found to have a "%s" method. Rather than the config file, this method will be used to determine the default extraction parameter and level.' % green(special_method)
                alert(msg,'scentry.learn_metadata')
            # Estimate a good extraction radius and level for an input scentry object from the BAM catalog
            this.default_extraction_par, this.default_level, this.extraction_map_dict = handler.__dict__[special_method](this)
            # NOTE: this.extraction_map_dict is a dictionary that contains two maps. extraction_map_dict['radius_map'] and extraction_map_dict['level_map']
            # extraction_map_dict['radius_map'] maps the extraction radius number to the extraction radius in units of M
            # extraction_map_dict['level_map'] maps the extraction radius number to appropriate level number. This one only applies to BAM.
        else:
            error('nrutils must be provided a infer_default_level_and_extraction_parameter method to map between extraction radii and extraction parameter. For some NR groups, eg GT, this map is trivial: extraction_parameter = extraction_radius. Data cannot be loaded unless this is sorted. See the bam or maya handler files for reference.')
            # NOTE that otherwise, values from the configuration file will be used
            this.default_extraction_par = this.config.default_par_list[0]
            this.default_level = this.config.default_par_list[1]
            this.extraction_map_dict = {}
            this.extraction_map_dict['radius_map'] = None
            this.extraction_map_dict['level_map'] = None
            # NOTE: this.extraction_map_dict is a dictionary that contains two maps. extraction_map_dict['radius_map'] and extraction_map_dict['level_map']
            # The default values are to set these to None.

        # Basic sanity check for standard attributes. NOTE this section needs to be completed and perhaps externalized to the current function.

        # Check that initial binary separation is float
        if not isinstance( this.b , float ) :
            msg = 'b = %g' % this.b
            raise ValueError(msg)
        # Check that final mass is float
        if not isinstance( this.mf , float ) :
            msg = 'final mass must be float, but %s found' % type(this.mf).__name__
            raise ValueError(msg)
        # Check that inital mass1 is float
        if not isinstance( this.m1 , float ) :
            msg = 'm1 must be float but %s found' % type(this.m1).__name__
            raise ValueError(msg)
        # Check that inital mass2 is float
        if not isinstance( this.m2 , float ) :
            msg = 'm2 must be float but %s found' % type(this.m2).__name__
            raise ValueError(msg)

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # Enfore m1>m2 convention.
        satisfies_massratio_convetion = lambda e: (not e.m1 > e.m2) and (not allclose(e.m1,e.m2,atol=1e-4))
        if satisfies_massratio_convetion(this):
            this.flip()
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        if satisfies_massratio_convetion(this):
            msg = 'Mass ratio convention m1>m2 must be used. Check scentry.flip(). It should have corrected this! \n>> m1 = %g, m2 = %g' % (this.m1,this.m2)
            raise ValueError(msg)

    # Create dynamic function that references the user's current configuration to construct the simulation directory of this run.
    def simdir(this):

        #
        import os

        if this.config:
            if 'static' in this.__dict__:
                if this.static:
                    ans = this.config.simdir
                else:
                    ans = this.config.reconfig().catalog_dir + this.relative_simdir
            else:
                ans = this.config.reconfig().catalog_dir + this.relative_simdir
            if not this.config.config_exists:
                msg = 'The current object has been marked as '+red('non-existent')+', likely by reconfig(). Please verify that the ini file for the related run exists. You may see this message for other (yet unpredicted) reasons.'
                error(msg,'scentry.simdir()')
        else:
            ans = '*'
        return ans

    # Flip 1->2 associations.
    def flip(this):

        #
        from numpy import array,double

        # Store the flippoed variables to placeholders
        R1 = array(this.R2); R2 = array(this.R1);
        m1 = double(this.m2); m2 = double(this.m1);
        P1 = array(this.P2); P2 = array(this.P1);
        L1 = array(this.L2); L2 = array(this.L1);
        S1 = array(this.S2); S2 = array(this.S1);
        X1 = array(this.X2); X2 = array(this.X1);

        # Apply the flip to the current object
        this.R1 = R1; this.R2 = R2
        this.m1 = m1; this.m2 = m2
        this.P1 = P1; this.P2 = P2
        this.L1 = L1; this.L2 = L2
        this.S1 = S1; this.S2 = S2
        this.X1 = X1; this.X2 = X2

    # Compare this scentry object to another using initial parameter fields. Return true false statement
    def compare2( this, that, atol=1e-3 ):

        #
        from numpy import allclose,hstack,double

        # Calculate an array of initial parameter values (the first element is 0 or 1 describing quasi-circularity)
        def param_array( entry ):

            # List of fields to add to array: initial parameters that are independent of initial separation
            field_list = [ 'm1', 'm2', 'S1', 'S2' ]

            #
            a = double( 'qc' in entry.label )
            for f in field_list:
                a = hstack( [a, entry.__dict__[f] ] )

            #
            return a

        # Perform comparison and return
        return allclose( param_array(this), param_array(that), atol=atol )


# Concert a simulation directory to a scentry object
# NOTE that this algorithm is to be compared to scbuild()
def simdir2scentry( catalog_dir, verbose = False ):

    # Load useful packages
    from commands import getstatusoutput as bash
    from os.path import realpath, abspath, join, splitext, basename
    from os import pardir,system,popen
    import pickle

    #
    if verbose: alert('We will now try to convert a given directory to a scentry object.', heading=True,pattern='--')

    # Load all known configs
    cpath_list = glob.glob( global_settings.config_path+'*.ini' )
    # Warn/Error if needed
    if not cpath_list:
        msg = 'Cannot find configuration files (*.ini) in %s' % global_settings.config_path
        error(msg,thisfun)

    # Create config objects from list of config files
    if verbose: alert('(1) The directory must be compatible with a known config ini file. Let\'s load all known configs and use the first one which successfully parses the given directory.',header=True,pattern='##')
    configs = [ scconfig( config_path, verbose=verbose ) for config_path in cpath_list ]

    # For each config, look for the related metadata file; stop after the first metadata file is found
    if verbose: alert('(2) For all configs, try to build a catalog object.',header=True,pattern='##')
    for config in configs:

        # NOTE: At this point the algorithm differs from scbuild

        # Set the catalog dir for this object to be the input, not what's in the config ini
        config.catalog_dir = catalog_dir
        config.simdir = catalog_dir

        # Search recurssively within the config's catalog_dir for files matching the config's metadata_id
        msg = 'Searching for %s in %s.' % ( cyan(config.metadata_id), cyan(catalog_dir) ) + yellow(' This may take a long time if the folder being searched is mounted from a remote drive.')
        if verbose: alert(msg,header=True)
        mdfile_list = rfind(catalog_dir,config.metadata_id,verbose=verbose)
        if verbose: alert('done.')

        # (try to) Create a catalog entry for each valid metadata file
        catalog = []
        h = -1
        for mdfile in mdfile_list:

            # Create tempoary scentry object
            entry = scentry(config,mdfile,static=True,verbose=verbose)

            # Set the simulation's set name to note that this entry is not a part of a catalog
            entry.setname = 'non-catalog'# entry.simname

            # If the obj is valid, add it to the catalog list, else ignore
            if entry.isvalid:
                catalog.append( entry )
            else:
                if verbose: print entry.log
                del entry

        # Break the for-loop if a valid scentry has been built
        if len( catalog ) > 0 : break

    #
    if len(catalog):
        if verbose: alert('(3) The previous config was able to parse the given directory --- We\'re all done here! :D',header=True,pattern='##')
    else:
        error( 'No configs were able to parse the given directory. Use this function with verbose=True to learn more.' )

    #
    return catalog


# Create the catalog database, and store it as a pickled file.
def scbuild(keyword=None,save=True):

    # Load useful packages
    from commands import getstatusoutput as bash
    from os.path import realpath, abspath, join, splitext, basename
    from os import pardir,system,popen
    import pickle

    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    # Look for config files
    cpath_list = glob.glob( global_settings.config_path+'*.ini' )

    # If a keyword is give, filter against found config files
    if isinstance(keyword,(str,unicode)):
        msg = 'Filtering ini files for \"%s\"'%cyan(keyword)
        alert(msg,'scbuild')
        cpath_list = filter( lambda path: keyword in path, cpath_list )

    #
    if not cpath_list:
        msg = 'Cannot find configuration files (*.ini) in %s' % global_settings.config_path
        error(msg,thisfun)

    # Create config objects from list of config files
    configs = [ scconfig( config_path ) for config_path in cpath_list ]

    # For earch config
    for config in configs:

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        # Create streaming log file        #
        logfstr = global_settings.database_path + '/' + splitext(basename(config.config_file_location))[0] + '.log'
        msg = 'Opening log file in: '+cyan(logfstr)
        alert(msg,thisfun)
        logfid = open(logfstr, 'w')
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

        # If catalog_dir is list of dirs, then select the first one that exists
        catalog_dir = config.catalog_dir
        if isinstance( config.catalog_dir, list ):
            warning('Multiple catalog directories found. We will scan through the related list, and then store first the catalog_dir that the OS can find.')
            for d in config.catalog_dir:
                from os.path import isdir
                if isdir(d):
                    catalog_dir = d
                    warning('Selecting "%s"'%cyan(d))
                    break

        # Search recurssively within the config's catalog_dir for files matching the config's metadata_id
        msg = 'Searching for %s in %s.' % ( cyan(config.metadata_id), cyan(catalog_dir) ) + yellow(' This may take a long time if the folder being searched is mounted from a remote drive.')
        alert(msg,thisfun)
        mdfile_list = rfind(catalog_dir,config.metadata_id,verbose=True)
        alert('done.',thisfun)

        # (try to) Create a catalog entry for each valid metadata file
        catalog = []
        h = -1
        for mdfile in mdfile_list:

            # Create tempoary scentry object
            # NOTE add code to look for previously processed scentry obejct in the same dir as the metadata file; if exists or not overwrite, load it; else create a new one
            entry = scentry(config,mdfile,verbose=True)

            # Write to the master log file
            h+=1
            logfid.write( '%5i\t%s\n'% (h,entry.log) )

            # If the obj is valid, add it to the catalog list, else ignore
            if entry.isvalid:
                catalog.append( entry )
                # NOTE: add code to store scentry object to metadata file location
            else:
                del entry

        # Store the catalog to the database_path
        if save:
            db = global_settings.database_path + '/' + splitext(basename(config.config_file_location))[0] + '.' + global_settings.database_ext
            msg = 'Saving database file to %s'%cyan(db)
            alert(msg,'scbuild')
            with open(db, 'wb') as dbf:
                pickle.dump( catalog , dbf, pickle.HIGHEST_PROTOCOL )

        # Close the log file
        logfid.close()

        #
        wave_train = ''#'~~~~<vvvvvvvvvvvvvWw>~~~~'
        hline = wave_train*3
        msg = '\n\n#%s#\n%s with \"%s\". The related log file is at \"%s\".\n#%s#'%(hline,hlblack('Done'),green(catalog_dir),green(logfstr),hline)
        alert(msg,'scbuild')


# Reconfigure a database file to the user's current nrutils configuration
def screconf( database_path, verbose=True, make_backup=True ):
    '''
    Reconfigure a database file to the user's current nrutils configuration. This function is to be used to assist users ability to apply third party catalog files to their nrutils installations
    '''
    # Import useful things
    from os import rename as fmove
    from shutil import copyfile as fcopy
    # Determine the database file name and parent directory
    if verbose: alert('Determining the database file name and parent directory')
    dbname = database_path.split('/')[-1]
    dbdir = '/'.join(database_path.split('/')[:-1])+'/'
    backup_database_path = dbdir+dbname.split('.')[0]+'.backup_db'
    # Make a backup of the database file
    if make_backup:
        if verbose: alert('Make a backup of the database file: %s'%cyan(backup_database_path))
        fcopy(database_path,backup_database_path)
    # Open the original file, and store contents
    if verbose: alert('Opening the original file ...')
    with open( database_path , 'rb') as dbf:
        catalog = pickle.load( dbf )
    if verbose: alert('Applying the granular reconfigure operation to the file\'s contents ...')
    # Apply the granular reconfigure operation to the file's contents
    for e in catalog:
        e.config.reconfig()
    # Save the reconfigured contents to the file, overwriting the original
    if verbose: alert('Saving the reconfigured contents to %s'%cyan(database_path))
    with open(database_path, 'wb') as dbf:
        pickle.dump( catalog , dbf, pickle.HIGHEST_PROTOCOL )


# Function for searching through catalog files.
def scsearch( catalog = None,           # Manually input list of scentry objects to search through
              q = None,                 # RANGE of mass ratios (>=1) to search for
              nonspinning = None,       # Non-spinning initially
              spinaligned = None,       # spin-aligned with L AND no in-plane spin INITIALLY
              spinantialigned = None,   # spin-anti-aligned with L AND no in-plane spin INITIALLY
              precessing = None,        # not spin aligned
              nonprecessing = None,     # not precessing
              equalspin = None,         # equal spin magnitudes
              unequalspin = None,       # not equal spin magnitudes
              antialigned = None,       # spin is in opposite direction of L
              setname = None,           # name of simulation set
              notsetname = None,        # list of setnames to ignore
              institute = None,         # list of institutes to accept
              keyword = None,           # list of keywords to accept (based on metadata directory string)
              notkeyword = None,        # list of keywords to not accept (based on metadata
                                        # directory string
              unique = None,            # if true, only simulations with unique initial conditions will be used
              plot = None,              # whether or not to show a plot of results
              exists=None,              # Test whether data directory related to scentry and ini file exist (True/False)
              validate_remnant=None,   # If true, ensure that final mass adn spin are well defined
              apply_remnant_fit=None,   # Toggle to apply model fit mapping initial to remnant params
              verbose = None):          # be verbose

    # Print non None inputs to screen
    thisfun = inspect.stack()[0][3]
    if verbose is not None:
        for k in dir():
            if (eval(k) is not None) and (k != 'thisfun'):
                # alert('Found %s keyword.' % (textul(k)) )
                alert('Found %s (=%s) keyword.' % (textul(k),str(eval(k) if not isinstance(eval(k),list) else '...' )) )

    '''
    Handle individual cases in serial
    '''

    #
    from os.path import realpath, abspath, join
    from os import pardir
    from numpy.linalg import norm
    from numpy import allclose,dot
    import pickle, glob

    # absolute tolerance for num comparisons
    tol = 1e-6

    # Handle the catalog input
    if catalog is None:
        # Get a list of all catalog database files. NOTE that .cat files are either placed in database_path directly, or by scbuild()
        dblist = glob.glob( global_settings.database_path+'*.'+global_settings.database_ext )
        # Let the people know which catalog files have been found
        if verbose==2:
            alert('Searching in "%s" for catalog files.'%yellow(global_settings.database_path))
        # Load the catalog file(s)
        catalog = []
        if verbose==2: alert('Loading catalog information from:')
        for db in dblist:
            if verbose==2: print '>> %s' % cyan(db)
            with open( db , 'rb') as dbf:
                catalog = catalog + pickle.load( dbf )

    # Determine whether remnant properties are already stored
    if validate_remnant is True:
        from numpy import isnan,sum
        test = lambda k: (sum(isnan( k.xf ))==0) and (isnan(k.mf)==0)
        catalog = filter( test, catalog )

    # mass-ratio
    qtol = 1e-3
    if q is not None:
        # handle int of float input
        if isinstance(q,(int,float)): q = [q-qtol,q+qtol]
        # NOTE: this could use error checking
        test = lambda k: k.m1/k.m2 >= min(q) and k.m1/k.m2 <= max(q)
        catalog = filter( test, catalog )

    # nonspinning
    if nonspinning is True:
        test = lambda k: norm(k.S1)+norm(k.S2) < tol
        catalog = filter( test, catalog )

    # spin aligned with orbital angular momentum
    if spinaligned is True:
        test = lambda k: allclose( dot(k.S1,k.L1+k.L2) , norm(k.S1)*norm(k.L1+k.L2) , atol=tol ) and allclose( dot(k.S2,k.L1+k.L2) , norm(k.S2)*norm(k.L1+k.L2) , atol=tol ) and not allclose( norm(k.S1)+norm(k.S2), 0.0, atol=tol )
        catalog = filter( test, catalog )

    # spin anti-aligned with orbital angular momentum
    if spinantialigned is True:
        test = lambda k: allclose( dot(k.S1,k.L1+k.L2) , -norm(k.S1)*norm(k.L1+k.L2) , atol=tol ) and allclose( dot(k.S2,k.L1+k.L2) , -norm(k.S2)*norm(k.L1+k.L2) , atol=tol ) and not allclose( norm(k.S1)+norm(k.S2), 0.0, atol=tol )
        catalog = filter( test, catalog )

    # precessing
    if precessing is True:
        test = lambda k: not allclose( abs(dot(k.S1+k.S2,k.L1+k.L2)), norm(k.L1+k.L2)*norm(k.S1+k.S2) , atol = tol )
        catalog = filter( test, catalog )

    # non-precessing, same as spinaligned & spin anti aligned
    nptol = 1e-4
    if nonprecessing is True:
        test = lambda k: allclose( abs(dot(k.S2,k.L1+k.L2)), norm(k.L1+k.L2)*norm(k.S2) , atol = nptol ) and allclose( abs(dot(k.S1,k.L1+k.L2)), norm(k.L1+k.L2)*norm(k.S1) , atol = nptol )
        catalog = filter( test, catalog )

    # spins have equal magnitude
    if equalspin is True:
        test = lambda k: allclose( k.S1, k.S2, atol = tol )
        catalog = filter( test, catalog )

    # spins have unequal magnitude
    if unequalspin is True:
        test = lambda k: not allclose( k.S1, k.S2, atol = tol )
        catalog = filter( test, catalog )

    #
    if antialigned is True:
        test = lambda k: allclose( dot(k.S1+k.S2,k.L1+k.L2)/(norm(k.S1+k.S2)*norm(k.L1+k.L2)), -1.0, atol = tol )
        catalog = filter( test, catalog )

    # Compare setname strings
    if setname is not None:
        if isinstance( setname, str ):
            setname = [setname]
        setname = filter( lambda s: isinstance(s,str), setname )
        setname = [ k.lower() for k in setname ]
        if isinstance( setname, list ) and len(setname)>0:
            test = lambda k: k.setname.lower() in setname
            catalog = filter( test, catalog )
        else:
            msg = 'setname input must be nonempty string or list.'
            error(msg)

    # Compare not setname strings
    if notsetname is not None:
        if isinstance( notsetname, str ):
            notsetname = [notsetname]
        notsetname = filter( lambda s: isinstance(s,str), notsetname )
        notsetname = [ k.lower() for k in notsetname ]
        if isinstance( notsetname, list ) and len(notsetname)>0:
            test = lambda k: not ( k.setname.lower() in notsetname )
            catalog = filter( test, catalog )
        else:
            msg = 'notsetname input must be nonempty string or list.'
            error(msg)

    # Compare institute strings
    if institute is not None:
        if isinstance( institute, str ):
            institute = [institute]
        institute = filter( lambda s: isinstance(s,str), institute )
        institute = [ k.lower() for k in institute ]
        if isinstance( institute, list ) and len(institute)>0:
            test = lambda k: k.config.institute.lower() in institute
            catalog = filter( test, catalog )
        else:
            msg = 'institute input must be nonempty string or list.'
            error(msg)

    # Compare keyword
    if keyword is not None:

        # If string, make list
        if isinstance( keyword, str ):
            keyword = [keyword]
        keyword = filter( lambda s: isinstance(s,str), keyword )

        # Determine whether to use AND or OR based on type
        if isinstance( keyword, list ):
            allkeys = True
            if verbose:
                msg = 'List of keywords or string keyword found: '+cyan('ALL scentry objects matching will be passed.')+' To pass ANY entries matching the keywords, input the keywords using an iterable of not of type list.'
                alert(msg)
        else:
            allkeys = False # NOTE that this means: ANY keys will be passed
            if verbose:
                msg = 'List of keywords found: '+cyan('ANY scentry objects matching will be passed.')+' To pass ALL entries matching the keywords, input the kwywords using a list object.'
                alert(msg)

        # Always lower
        keyword = [ k.lower() for k in keyword ]

        # Handle two cases
        if allkeys:
            # Treat different keys with AND
            for key in keyword:
                test = lambda k: key in k.metadata_file_location.lower()
                catalog = filter( test, catalog )
        else:
            # Treat different keys with OR
            temp_catalogs = [ catalog for w in keyword ]
            new_catalog = []
            for j,key in enumerate(keyword):
                test = lambda k: key in k.metadata_file_location.lower()
                new_catalog += filter( test, temp_catalogs[j] )
            catalog = list(set(new_catalog))

    # Compare not keyword
    if notkeyword is not None:
        if isinstance( notkeyword, str ):
            notkeyword = [notkeyword]
        notkeyword = filter( lambda s: isinstance(s,str), notkeyword )
        notkeyword = [ k.lower() for k in notkeyword ]
        for w in notkeyword:
            test = lambda k: not ( w in k.metadata_file_location.lower() )
            catalog = filter( test, catalog )

    # Validate the existance of the related config files and simulation directories
    # NOTE that this effectively requires two reconfigure instances and is surely suboptimal
    if not ( exists is None ):
        def isondisk(e):
            ans = (e.config).reconfig().config_exists and os.path.isdir(e.simdir())
            if not ans:
                msg = 'Ignoring entry at %s becuase its config file cannot be found and/or its simulation directory cannot be found.' % cyan(e.simdir())
                warning(msg)
            return ans
        if catalog is not None:
            catalog = filter( isondisk , catalog )

    # Filter out physically degenerate simuations within a default tolerance
    output_descriptor = magenta(' possibly degenerate')
    if unique:
        catalog = scunique(catalog,verbose=False)
        output_descriptor = green(' unique')

    # Sort by date
    catalog = sorted( catalog, key = lambda e: e.date_number, reverse = True )

    #
    if apply_remnant_fit:
        #
        from numpy import array
        # Let the people know
        if verbose:
            warning('Applying remant fit to scentry objects. This should be done if the final mass and spin meta data are not trustworth. '+magenta('The fit being used only works for non-precessing systems.'))
        #
        for e in catalog:
            #e.mf,e.xf = Mf14067295(e.m1,e.m2,e.X1[-1],e.X2[-1]),jf14067295(e.m1,e.m2,e.X1[-1],e.X2[-1])
            e.mf,e.xf = remnant(e.m1,e.m2,e.X1[-1],e.X2[-1])
            e.Sf = e.mf*e.mf*array([0,0,e.xf])
            e.Xf = e.Sf/(e.mf**2)

    #
    if verbose:
        if len(catalog)>0:
            print '## Found %s%s simulations:' % ( bold(str(len(catalog))), output_descriptor )
            for k,entry in enumerate(catalog):
                # tag this entry with its inferred simname
                simname = entry.raw_metadata.source_dir[-1].split('/')[-1] if entry.raw_metadata.source_dir[-1][-1]!='/' else entry.raw_metadata.source_dir[-1].split('/')[-2]
                print '[%04i][%s] %s: %s\t(%s)' % ( k+1, green(entry.config.config_file_location.split('/')[-1].split('.')[0]), cyan(entry.setname), entry.label, cyan(simname ) )
        else:
            warning('!! Found %s simulations.' % str(len(catalog)))
        print ''

    #
    return catalog



# Given list of scentry objects, make a list unique in initial parameters
def scunique( catalog = None, tol = 1e-3, verbose = False ):

    # import useful things
    from numpy import ones,argmax,array

    # This mask will be augmented such that only unique indeces are true
    umap = ones( len(catalog), dtype=bool )

    # Keep track of which items have been compared using another map
    tested_map = ones( len(catalog), dtype=bool )

    # For each entry in catalog
    for d,entry in enumerate(catalog):

        #
        if tested_map[d]:

            # Let the people know.
            if verbose:
                alert( '[%i] %s:%s' % (d,entry.setname,entry.label), 'scunique' )

            # Create a map of all simulations with matching initial parameters (independently of initial setaration)

            # 1. Filter out all matching objects. NOTE that this subset include the current object
            subset = filter( lambda k: entry.compare2(k,atol=tol), catalog )

            # 2. Find index locations of subset
            subdex = [ catalog.index(k) for k in subset ]

            # 3. By default, select longest run to keep. maxdex is the index in subset where b takes on its largest value.
            maxdex = argmax( [ e.b for e in subset ] ) # recall that b is initial separation

            # Let the people know.
            for ind,k in enumerate(subset):
                tested_map[ subdex[ind] ] = False
                if k is subset[maxdex]:
                    if verbose: print '>> Keeping: [%i] %s:%s' % (catalog.index(k),k.setname,k.label)
                else:
                    umap[ subdex[ind] ] = False
                    if verbose: print '## Removing:[%i] %s:%s' % (catalog.index(k),k.setname,k.label)

        else:

            if verbose: print magenta('[%i] Skipping %s:%s. It has already been checked.' % (d,entry.setname,entry.label) )

    # Create the unique catalog using umap
    unique_catalog = list( array(catalog)[ umap ] )

    # Let the people know.
    if verbose:
        print green('Note that %i physically degenerate simulations were removed.' % (len(catalog)-len(unique_catalog)) )
        print green( 'Now %i physically unique entries remain:' % len(unique_catalog) )
        for k,entry in enumerate(unique_catalog):
            print green( '>> [%i] %s: %s' % ( k+1, entry.setname, entry.label ) )
        print ''

    # return the unique subset of runs
    return unique_catalog



# Construct string label for members of the scentry class
def sclabel( entry,             # scentry object
             use_q = True ):    # if True, mass ratio will be used in the label

    #
    def sclabel_many( entry = None, use_q = None ):

        #
        from numpy import sign

        #
        tag_list = []
        for e in entry:
            # _,tg = sclabel_single( entry = e, use_q = use_q )
            tg = e.label.split('-')
            tag_list.append(tg)

        #
        common_tag_set = set(tag_list[0])
        for k in range(2,len(tag_list)):
            common_tag_set &= set(tag_list[k])

        #
        common_tag = [ k for k in tag_list[0] if k in common_tag_set ]

        #
        single_q = False
        for tg in common_tag:
            single_q = single_q or ( ('q' in tg) and (tg!='qc') )

        #
        tag = common_tag

        #
        if not single_q:
            tag .append('vq')   # variable q

        # concat tags together to make label
        label = ''
        for k in range(len(tag)):
            label += sign(k)*'-' + tag[k]

        #
        return label


    #
    def sclabel_single( entry = None, use_q = None ):

        #
        from numpy.linalg import norm
        from numpy import allclose,dot,sign

        #
        if not isinstance( entry, scentry ):
            msg = '(!!) First input must be member of scentry class.'
            raise ValueError(msg)

        # Initiate list to hold label parts
        tag = []

        #
        tol = 1e-4

        # shorthand for entry
        e = entry

        # Calculate the entry's net spin and oribal angular momentum
        S = e.S1+e.S2; L = e.L1+e.L2

        # Run is quasi-circular if momenta are perpindicular to separation vector
        R = e.R2 - e.R1
        if allclose( dot(e.P1,R), 0.0 , atol=tol ) and allclose( dot(e.P2,R), 0.0 , atol=tol ):
            tag.append('qc')

        # Run is nonspinning if both spin magnitudes are close to zero
        if allclose( norm(e.S1) + norm(e.S2) , 0.0 , atol=tol ):
            tag.append('ns')

        # Label by spin on BH1 if spinning
        if not allclose( norm(e.S1), 0.0, atol=tol ) :
            tag.append( '1chi%1.2f' % ( norm(e.S1)/e.m1**2 ) )

        # Label by spin on BH2 if spinning
        if not allclose( norm(e.S2), 0.0, atol=tol ) :
            tag.append( '2chi%1.2f' % ( norm(e.S2)/e.m2**2 ) )

        # Run is spin aligned if net spin is parallel to net L
        if allclose( dot(e.S1,L) , norm(e.S1)*norm(L) , atol=tol ) and allclose( dot(e.S2,L) , norm(e.S2)*norm(L) , atol=tol ) and (not 'ns' in tag):
            tag.append('sa')

        # Run is spin anti-aligned if net spin is anti-parallel to net L
        if allclose( dot(e.S1,L) , -norm(e.S1)*norm(L) , atol=tol ) and allclose( dot(e.S2,L) , -norm(e.S2)*norm(L) , atol=tol ) and (not 'ns' in tag):
            tag.append('saa')

        # Run is precessing if component spins are not parallel with L
        if (not 'sa' in tag) and (not 'saa' in tag) and (not 'ns' in tag):
            tag.append('p')

        # mass ratio
        if use_q:
            tag.append( 'q%1.2f' % (e.m1/e.m2) )

        # concat tags together to make label
        label = ''
        for k in range(len(tag)):
            label += sign(k)*'-' + tag[k]

        #
        return label, tag

    #
    if isinstance( entry, list ):
       label = sclabel_many( entry = entry, use_q = use_q )
    elif isinstance( entry, scentry ):
       label,_ = sclabel_single( entry = entry, use_q = use_q )
    else:
       msg = 'input must be list scentry objects, or single scentry'
       raise ValueError(msg)

    #
    return label


# Lowest level class for gravitational waveform data
class gwf:

    # Class constructor
    def __init__( this,                         # The object to be created
                  wfarr=None,                   # umpy array of waveform data in to format [time plus imaginary]
                  dt                    = None, # If given, the waveform array will be interpolated to this
                                                # timestep if needed
                  ref_scentry           = None, # reference scentry object
                  l                     = None, # Optional polar index (an eigenvalue of a differential eq)
                  m                     = None, # Optional azimuthal index (an eigenvalue of a differential eq)
                  extraction_parameter  = None, # Optional extraction parameter ( a map to an extraction radius )
                  kind                  = None, # strain or psi4
                  friend                = None, # gwf object from which to clone fields
                  mf                    = None, # Optional remnant mass input
                  xf                    = None, # Optional remnant spin input
                  m1=None,m2=None,              # Optional masses
                  label                 = None, # Optional label input (see gwylm)
                  preinspiral           = None, # Holder for information about the raw waveform's turn-on
                  postringdown          = None, # Holder for information about the raw waveform's turn-off
                  verbose = False ):    # Verbosity toggle

        #
        this.dt = dt

        # The kind of obejct to be created : e.g. psi4 or strain
        if kind is None:
            kind = r'$y$'
        this.kind = kind

        # Optional field to be set externally if needed
        source_location = None

        # Set optional fields to none as default. These will be set externally is they are of use.
        this.l = l
        this.m = m
        this.extraction_parameter = extraction_parameter

        #
        this.verbose = verbose

        # Fix nans, nonmonotinicities and jumps in time series waveform array
        wfarr = straighten_wfarr( wfarr, verbose=this.verbose )

        # use the raw waveform data to define all fields
        this.wfarr = wfarr

        # optional component masses
        this.m1,this.m2 = m1,m2

        # Optional Holders for remnant mass and spin
        this.mf = mf
        this.xf = xf

        # Optional label input (see gwylm)
        this.label = label

        #
        this.preinspiral = preinspiral
        this.postringdown = postringdown

        #
        this.ref_scentry = ref_scentry

        this.setfields(wfarr=wfarr,dt=dt)

        # If desired, Copy fields from related gwf object.
        if type(friend).__name__ == 'gwf' :
            this.meet( friend )
        elif friend is not None:
            msg = 'value of "friend" keyword must be a member of the gwf class'
            error(mgs,'gwf')

        # Store wfarr in a field that will not be touched beyond this point. This is useful because
        # the properties defined in "setfields" may change as the waveform is manipulated (e.g. windowed,
        # scaled, phase shifted), and after any of these changes, we may want to reaccess the initial waveform
        # though the "reset" method (i.e. this.reset)
        this.__rawgwfarr__ = wfarr

        # Tag for whether the wavform has been low pass filtered since creation
        this.__lowpassfiltered__ = False

    # set fields of standard wf object
    def setfields(this,         # The current object
                  wfarr=None,   # The waveform array to apply to the current object
                  setfd=True,   # Option to toggle freq domain calculations
                  dt=None):     # The time spacing to apply to the current object

        # Alert the use if improper input is given
        if (wfarr is None) and (this.wfarr is None):
            msg = 'waveform array input (wfarr=) must be given'
            raise ValueError(msg)
        elif wfarr is not None:
            this.wfarr = wfarr
        elif (wfarr is None) and not (this.wfarr is None):
            wfarr = this.wfarr
        else:
            msg = 'unhandled waveform array configuration: input wfarr is %s and this.wfarr is %s'%(wfarr,this.wfarr)
            error(msg,'gwf.setfields')

        # If given dt, then interpolote waveform array accordingly
        wfarr = straighten_wfarr( wfarr, this.verbose )
        wfdt = wfarr[1,0]-wfarr[0,0]

        from numpy import abs
        if (dt is not None) and (abs(dt-wfdt)/(dt+wfdt)>1e-6):
          if this.verbose:
              msg = 'Interpolating data to '+cyan('dt=%f'%dt)
              alert(msg)
          wfarr = intrp_wfarr(wfarr,delta=dt)

        ##########################################################
        # Make sure that waveform array is in t-plus-cross format #
        ##########################################################

        # Imports
        from numpy import abs,sign,linspace,exp,arange,angle,diff,ones,isnan,pi,log
        from numpy import vstack,sqrt,unwrap,arctan,argmax,mod,floor,logical_not
        from scipy.interpolate import InterpolatedUnivariateSpline
        from scipy.fftpack import fft, fftfreq, fftshift, ifft

        # Time domain attributes
        this.t          = None      # Time vals
        this.plus       = None      # Plus part
        this.cross      = None      # Cross part
        this.y          = None      # Complex =(def) plus + 1j*cross
        this.amp        = None      # Amplitude = abs(y)
        this.phi        = None      # Complex argument
        this.dphi       = None      # Time rate of complex argument
        this.k_amp_max  = None      # Index location of amplitude max
        this.window     = None      # The time domain window function applid to the original waveform. This
                                    # initiated as all ones, but changed in the taper method (if it is called)

        # Frequency domain attributes. NOTE that this will not currently be set by default.
        # Instead, the current approach will be to set these fields once gwf.fft() has been called.
        this.f              = None      # double sided frequency range
        this.w              = None      # double sided angular frequency range
        this.fd_plus        = None      # fourier transform of time domain plus part
        this.fd_cross       = None      # fourier transform of time domain cross part
        this.fd_y           = None      # both polarisations (i.e. plus + ij*cross)
        this.fd_wfarr       = None      # frequency domain waveform array
        this.fd_amp         = None      # total frequency domain amplitude: abs(right+left)
        this.fd_phi         = None      # total frequency domain phase: arg(right+left)
        this.fd_dphi        = None      # frequency derivative of fdphi
        this.fd_k_amp_max   = None      # index location of fd amplitude max

        # Domain independent attributes
        this.n              = None      # length of arrays
        this.fs             = None      # samples per unit time
        this.df             = None      # frequnecy domain spacing

        # Validate time step. Interpolate for constant time steo if needed.
        this.__validatet__()
        # Determine formatting of wfarr
        t = this.wfarr[:,0]; A = this.wfarr[:,1]; B = this.wfarr[:,2];

        # if all elements of A are greater than zero
        if (A>0).all() :
            typ = 'amp-phase'
        elif ((abs(A.imag)>0).any() or (abs(B.imag)>0).any()): # else if A or B are complex
            #
            msg = 'The current code version only works with plus valued time domain inputs to gwf().'
            raise ValueError(msg)
        else:
            typ = 'plus-imag'
        # from here on, we are to work with the plus-cross format
        if typ == 'amp-phase':
            C = A*exp(1j*B)
            this.wfarr = vstack( [ t, C.real, C.imag ] ).T
            this.__validatewfarr__()

        # --------------------------------------------------- #
        # Set time domain properties
        # --------------------------------------------------- #

        # NOTE that it will always be assumed that the complex waveform is plus+j*imag

        # Here, we trust the user to know that if one of these quantities is changed, then it will affect the other, and
        # that to have all quantities consistent, then one should modify wfarr, and then perform this.setfields()
        # (and not modify e.g. amp and phase). All functions on gwf objects will respect this.

        # Time domain attributed
        this.t      = this.wfarr[:,0]                               # Time
        this.plus   = this.wfarr[:,1]                               # Real part
        this.cross  = this.wfarr[:,2]                               # Imaginary part
        this.y      = this.plus + 1j*this.cross                     # Complex waveform
        this.amp    = abs( this.y )                                 # Amplitude

        phi_    = unwrap( angle( this.y ) )                         # Phase: NOTE, here we make the phase constant where the amplitude is zero
        # print find( (this.amp > 0) * (this.amp<max(this.amp)) )
        # k = find( (this.amp > 0) * (this.amp<max(this.amp)) )[0]
        # phi_[0:k] = phi_[k]
        this.phi = phi_

        this.dphi   = intrp_diff( this.t, this.phi )                # Derivative of phase, last point interpolated to preserve length
        # this.dphi   = diff( this.phi )/this.dt                # Derivative of phase, last point interpolated to preserve length


        # Only use the smooth part of the amplitude to determine the peak location. This is needed whenthe junk radiation has a higher amplitude than the physical peak.
        #mask = smoothest_part( this.amp )
        #this.k_amp_max = argmax( this.amp[mask[0]:] ) + mask[0]
        # this.k_amp_max = argmax( this.m * this.dphi * this.amp )
        # this.k_amp_max = argmax( this.amp )
        # print this.l, this.m
        this.k_amp_max = find_amp_peak_index( this.t, this.amp, plot=False ) # function lives in basics
        # this.k_amp_max = argmax(this.amp)                           # index location of max ampitude


        # Estimate true time location of peak amplitude
        try:
            # NOTE that teh k_amp_ax above is used to help intrp_t_amp_max
            this.intrp_t_amp_max = intrp_argmax(clean_amp,domain=this.t,ref_index=this.k_amp_max) # Interpolated time coordinate of max
        except:
            this.intrp_t_amp_max = this.t[this.k_amp_max]

        # # If the junk radiation has a higher amplitude than the "merger", then we must take care:
        # # NOTE, here will use a simple heuristic
        # start_chunk = this.t[find(this.amp>0)[0]] + 200 # (M)
        # __t__ = this.t
        # __t__ -= __t__[0]
        # __dt__ = __t__[1]-__t__[0]
        # __k__ = int( start_chunk/__dt__ )
        #
        # try:
        #     if max(this.amp[:__k__]) > max(this.amp[__k__:]):
        #         warning(red('The junk radiation appears to have a larger amplitude than that of merger, so we are going to be careful about how we characterize where the peak is.'))
        #         if diff(lim(this.t))>start_chunk:
        #             this.intrp_t_amp_max = this.t[__k__] + intrp_argmax( this.amp[(__k__+1):], domain=this.t )
        #             this.k_amp_max = __k__ + argmax(this.amp[ __k__+1 : ])
        # except:
        #     _ = _


        # ## Diagnostic code
        # print '** ', __dt__, max(this.amp[:__k__]), max(this.amp[__k__:])
        # from matplotlib.pyplot import plot,show,xlim
        # plot( this.t[:__k__],this.amp[:__k__] )
        # plot( this.t[__k__:],this.amp[__k__:] )
        # xlim([0,2*start_chunk])
        # show()

        #
        this.n      = len(this.t)                                   # Number of time samples
        this.window = ones( this.n )                                # initial state of time domain window
        this.fs     = 1.0/this.dt                                   # Sampling rate

        # --------------------------------------------------- #
        # Always calculate frequency domain data
        # --------------------------------------------------- #
        if setfd:

            # compute the frequency domain

            # fftlen = this.n if this.fftfactor is None else int( 2 ** ( int(log( this.n )/log(2)) + 1.0 + this.fftfactor ) )-1
            this.f = fftshift(fftfreq( this.n, this.dt ))

            this.w = 2*pi*this.f
            this.df     = this.f[1]-this.f[0]                                # freq resolution

            # compute fourier transform values
            this.fd_plus   = fftshift(fft( this.plus  )) * this.dt                    # fft of plus
            this.fd_cross  = fftshift(fft( this.cross )) * this.dt                    # fft of cross
            this.fd_y       = this.fd_plus + 1j*this.fd_cross               # full fft
            this.fd_amp     = abs( this.fd_y )                              # amp of full fft
            this.fd_phi     = unwrap( angle( this.fd_y ) )                  # phase of full fft

            # use a length preserving derivative
            this.fd_dphi    = intrp_diff( this.f, this.fd_phi )             # phase rate: dphi/df

            this.fd_k_amp_max = argmax( this.fd_amp )

            this.fd_wfarr = vstack( [this.f,this.fd_plus,this.fd_cross] ).T

        # Starting frequency in rad/sec
        this.wstart = None

    # Copy attrributed from friend.
    def meet(this,friend,init=False,verbose=False):

        # If wrong type input, let the people know.
        if friend.__class__.__name__!='gwf':
            msg = '1st input must be of type ' + bold(type(this).__name__)+'.'
            error( msg, fname=inspect.stack()[0][3] )

        # Define transferable attributes
        traits = ['ref_scentry', 'xf', 'label', 'm1', 'm2', 'extraction_parameter', 'preinspiral', 'kind', 'mf', 'postringdown', 'm', 'l']

        # Copy attrributed from friend. If init, then do not check if attribute already exists in this.
        for attr in traits:
            if verbose: print '\t that.%s --> this.%s (%s)' % (attr,attr,type(friend.__dict__[attr]).__name__)
            setattr( this, attr, friend.__dict__[attr] )

        #
        return this

    # validate whether there is a constant time step
    def __validatet__(this):
        #
        from numpy import diff,var,allclose,vstack,mean,linspace,diff,amin,allclose
        from numpy import arange,array,double,isnan,nan,logical_not,hstack
        from scipy.interpolate import InterpolatedUnivariateSpline

        # # Look for and remove nans
        # t,A,B = this.wfarr[:,0],this.wfarr[:,1],this.wfarr[:,2]
        # nan_mask = logical_not( isnan(t) ) * logical_not( isnan(A) ) * logical_not( isnan(B) )
        # if logical_not(nan_mask).any():
        #     msg = red('There are NANs in the data which mill be masked away.')
        #     warning(msg,'gwf.setfields')
        #     this.wfarr = this.wfarr[nan_mask,:]
        #     t = this.wfarr[:,0]; A = this.wfarr[:,1]; B = this.wfarr[:,2];

        # Note the shape convention
        t = this.wfarr[:,0]

        # check whether t is monotonically increasing
        isincreasing = allclose( t, sorted(t), 1e-6 )
        if not isincreasing:
            # Let the people know
            msg = red('The time series has been found to be non-monotonic. We will sort the data to enforce monotinicity.')
            warning(msg,'gwf.__validatet__')
            # In this case, we must sort the data and time array
            map_ = arange( len(t) )
            map_ = sorted( map_, key = lambda x: t[x] )
            this.wfarr = this.wfarr[ map_, : ]
            t = this.wfarr[:,0]

        # Look for duplicate time data
        hasduplicates = 0 == amin( diff(t) )
        if hasduplicates:
            # Let the people know
            msg = red('The time series has been found to have duplicate data. We will delete the corresponding rows.')
            warning(msg,'gwf.__validatet__')
            # delete the offending rows
            dup_mask = hstack( [True, diff(t)!=0] )
            this.wfarr = this.wfarr[dup_mask,:]
            t = this.wfarr[:,0]

        # if there is a non-uniform timestep, or if the input dt is not None and not equal to the given dt
        NONUNIFORMT = not isunispaced(t)
        INPUTDTNOTGIVENDT = this.dt is None
        proceed = False
        if NONUNIFORMT and (not INPUTDTNOTGIVENDT):
            msg = '(**) Waveform not uniform in time-step. Interpolation will be applied.'
            if this.verbose: print magenta(msg)
            print 'maxdt = '+str(diff(t).max())
            # proceed = True
        if (NONUNIFORMT and INPUTDTNOTGIVENDT) or proceed:
            # if dt is not defined and not none, assume smallest dt
            if this.dt is None:
                this.dt = diff(lim(t))/len(t)
                msg = '(**) Warning: No dt given to gwf(). We will assume that the input waveform array is in geometric units, and that dt = %g will more than suffice.' % this.dt
                if this.verbose:
                    print magenta(msg)
            # Interpolate waveform array
            intrp_t = arange( min(t), max(t), this.dt )
            intrp_R = InterpolatedUnivariateSpline( t, this.wfarr[:,1] )( intrp_t )
            intrp_I = InterpolatedUnivariateSpline( t, this.wfarr[:,2] )( intrp_t )
            # create final waveform array
            this.wfarr = vstack([intrp_t,intrp_R,intrp_I]).T
        else:
            # otherwise, set dt automatically
            this.dt = mean(diff(t))

    # validate shape of waveform array
    def __validatewfarr__(this):
        # check shape width
        if this.wfarr.shape[-1] != 3 :
            msg = '(!!) Waveform arr should have 3 columns'
            raise ValueError(msg)
        # check shape depth
        if len(this.wfarr.shape) != 2 :
            msg = '(!!) Waveform array should have two dimensions'
            raise ValueError(msg)

    # General plotting
    def plot( this,
              show=False,
              fig = None,
              title = None,
              ref_gwf = None,
              labels = None,
              tlim = None,
              sizescale=1.1,
              flim = None,
              ax=None,
              domain = None):

        # Handle which default domain to plot
        if domain is None:
            domain = 'time'
        elif not ( domain in ['time','freq'] ):
            msg = 'Error: domain keyword must be either "%s" or "%s".' % (cyan('time'),cyan('freq'))
            error(msg,'gwylm.plot')

        # Plot selected domain.
        if domain in ('time','t'):
            ax = this.plottd( show=show,fig=fig,title=title, ref_gwf=ref_gwf, labels=labels, tlim=tlim, sizescale=sizescale, ax=ax )
        elif domain in ('f','freq'):
            ax = this.plotfd( show=show,fig=fig,title=title, ref_gwf=ref_gwf, labels=labels, flim=flim, sizescale=sizescale, ax=ax )
        else:
            error('Domain must be in ("time","t","f","freq")')

        #
        from matplotlib.pyplot import gcf

        #
        return ax,gcf()

    # Plot frequency domain
    def plotfd( this,
                show    =   False,
                fig     =   None,
                title   =   None,
                ref_gwf = None,
                labels = None,
                flim = None,
                sizescale=1.1,
                ax=None,
                verbose =   False ):

        #
        from matplotlib.pyplot import plot,subplot,figure,tick_params,subplots_adjust,ylim
        from matplotlib.pyplot import grid,setp,tight_layout,margins,xlabel,legend,sca
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import ylabel as yl
        from matplotlib.pyplot import title as ttl
        from numpy import ones,sqrt,hstack,array,sign

        #from matplotlib import rc
        #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        #rc('text', usetex=True)

        #
        if ref_gwf:
            that = ref_gwf

        #
        if fig is None:
            fig = figure(figsize = sizescale*array([8,7.2]))
            fig.set_facecolor("white")

        #
        if ax is None:
            ax = []
            ax.append( subplot(3,1,1) )
            ax.append( subplot(3,1,2, sharex=ax[0]) )
            ax.append( subplot(3,1,3, sharex=ax[0]) )

        #
        kind = this.kind

        #
        clr = rgb(3)
        grey = 0.9*ones(3)
        lwid = 1
        txclr = 'k'
        fs = 18
        font_family = 'serif'
        gclr = '0.9'

        #
        # ax = []
        # xlim = lim(this.t) # [-400,this.t[-1]]

        # NOTE that m<0 cases are handled here via sign(m)
        pos_mask = sign(this.m if this.m is not None else 1)*this.f>0
        if ref_gwf:
            that_pos_mask = sign(that.m)*that.f>0
            that_lwid = 4
            that_alpha = 0.22

        #
        set_legend = False
        if not labels:
            labels = ('','')
        else:
            set_legend=True

        # ------------------------------------------------------------------- #
        # Amplitude
        # ------------------------------------------------------------------- #
        sca( ax[0] )

        grid(color=gclr, linestyle='-')
        setp(ax[0].get_xticklabels(), visible=False)
        ax[0].set_xscale('log', nonposx='clip')
        ax[0].set_yscale('log', nonposy='clip')
        #
        plot( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_amp[pos_mask], color=clr[0], label=labels[0] )
        if ref_gwf:
            plot( sign(that.m if that.m is not None else 1)*that.f[that_pos_mask], that.fd_amp[that_pos_mask], color=clr[0], linewidth=that_lwid, alpha=that_alpha, label=labels[-1] )
        # if this.m!=0: pylim( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_amp[pos_mask], pad_y=10 )

        # impose flim if its not None
        if flim is not None:
            mask = (this.f>min(flim)) & (this.f<max(flim))
            ylim( lim(this.fd_amp[pos_mask & mask],dilate=0.2) )

        #
        yl('$|$'+kind+'$|(f)$',fontsize=fs,color=txclr, family=font_family )
        if set_legend: legend(frameon=False)

        # ------------------------------------------------------------------- #
        # Total Phase
        # ------------------------------------------------------------------- #
        sca( ax[1] )
        grid(color=gclr, linestyle='-')
        setp(ax[1].get_xticklabels(), visible=False)
        ax[1].set_xscale('log', nonposx='clip')
        #
        plot( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_phi[pos_mask], color=1-clr[0] )
        if ref_gwf:
            plot( sign(that.m if that.m is not None else 1)*that.f[that_pos_mask], that.fd_phi[that_pos_mask], color=1-clr[0], linewidth=that_lwid, alpha=that_alpha )
        # if this.m!=0: pylim( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_phi[pos_mask] )
        #
        yl(r'$\phi = \mathrm{arg}($'+kind+'$)$',fontsize=fs,color=txclr, family=font_family )

        # impose flim if its not None
        if flim is not None:
            mask = (this.f>min(flim)) & (this.f<max(flim))
            ylim( lim(this.fd_phi[pos_mask & mask],dilate=0.2) )

        # ------------------------------------------------------------------- #
        # Total Phase Rate
        # ------------------------------------------------------------------- #
        sca( ax[2] )
        grid(color=gclr, linestyle='-')
        ax[2].set_xscale('log', nonposx='clip')
        #
        plot( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_dphi[pos_mask], color=sqrt(clr[0]) )
        if ref_gwf:
            plot( sign(that.m if that.m is not None else 1)*that.f[that_pos_mask], that.fd_dphi[that_pos_mask], color=sqrt(clr[0]), linewidth=that_lwid, alpha=that_alpha )
        # if this.m!=0: pylim( sign(this.m if this.m is not None else 1)*this.f[pos_mask], this.fd_dphi[pos_mask] )
        #
        yl(r'$\mathrm{d}{\phi}/\mathrm{d}f$',fontsize=fs,color=txclr, family=font_family)

        # impose flim if its not None
        if flim is not None:
            mask = (this.f>min(flim)) & (this.f<max(flim))
            ylim( lim(this.fd_dphi[pos_mask & mask],dilate=0.2) )

        # ------------------------------------------------------------------- #
        # Full figure settings
        # ------------------------------------------------------------------- #

        # impose flim if its not None
        if flim is not None:
            ax[0].set_xlim(flim)

        if title is not None:
            ax[0].set_title( title, family=font_family )

        # Set axis lines (e.g. grid lines) below plot lines
        for a in ax:
            a.set_axisbelow(True)

        # Ignore renderer warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight_layout(pad=2, w_pad=1.2)
            subplots_adjust(hspace = .001)

        #
        xlabel( r'$f$' if sign(this.m if this.m is not None else 1)>0 else r'$-f$' ,fontsize=fs,color=txclr)

        #
        if show:
            shw()

        #
        return ax

    # Plot time domain
    def plottd( this,
              show=False,
              fig = None,
              ref_gwf = None,
              labels = None,
              tlim = None,
              sizescale=1.1,
              ax=None,
              title = None):

        #
        import warnings
        from numpy import array

        #
        from matplotlib.pyplot import plot,subplot,figure,tick_params,subplots_adjust,sca
        from matplotlib.pyplot import grid,setp,tight_layout,margins,xlabel,legend,ylim,xlim
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import ylabel as yl
        from matplotlib.pyplot import title as ttl
        from numpy import ones,sqrt,hstack

        #
        if fig is None:
            fig = figure(figsize = sizescale*array([8,7.2]))
            fig.set_facecolor("white")

        #
        if ax is None:
            ax = []
            ax.append( subplot(3,1,1) )
            ax.append( subplot(3,1,2, sharex=ax[0]) )
            ax.append( subplot(3,1,3, sharex=ax[0]) )

        #
        clr = rgb(3)
        grey = 0.9*ones(3)
        lwid = 1
        txclr = 'k'
        fs = 18
        font_family = 'serif'
        gclr = '0.9'

        #
        if ref_gwf:
            that = ref_gwf
            that_lwid = 4
            that_alpha = 0.22

        #
        set_legend = False
        if not labels:
            labels = ('','')
        else:
            set_legend=True

        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        # Time domain plus and cross parts
        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        sca( ax[0] )
        grid(color=gclr, linestyle='-')
        setp(ax[0].get_xticklabels(), visible=False)
        # actual plotting
        plot( this.t, this.plus,  linewidth=lwid, color=0.8*grey )
        plot( this.t, this.cross, linewidth=lwid, color=0.5*grey )
        plot( this.t, this.amp,   linewidth=lwid, color=clr[0], label=labels[0] )
        plot( this.t,-this.amp,   linewidth=lwid, color=clr[0] )
        if ref_gwf:
            plot( that.t, that.plus,  linewidth=that_lwid, color=0.8*grey, alpha=that_alpha )
            plot( that.t, that.cross, linewidth=that_lwid, color=0.5*grey, alpha=that_alpha )
            plot( that.t, that.amp,   linewidth=that_lwid, color=clr[0], alpha=that_alpha, label=labels[-1] )
            plot( that.t,-that.amp,   linewidth=that_lwid, color=clr[0], alpha=that_alpha )
        if set_legend: legend(frameon=False)

        # Ignore renderer warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight_layout(pad=2, w_pad=1.2)
            subplots_adjust(hspace = .001)

        #
        #pylim( this.t if tlim is None else this.t[this.t>min(tlim) & this.t<max(tlim)] , this.amp, domain=xlim, symmetric=True )
        if tlim is not None:
            mask = (this.t>min(tlim)) & (this.t<max(tlim)) & (this.amp>0)
            ylim( array([-1,1])*this.amp[mask].max()*1.15 )

        kind = this.kind
        yl(kind,fontsize=fs,color=txclr, family=font_family )

        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        # Time domain phase
        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        sca( ax[1] )
        grid(color=gclr, linestyle='-')
        setp(ax[1].get_xticklabels(), visible=False)
        # actual plotting
        plot( this.t, this.phi, linewidth=lwid, color=1-clr[0] )
        if ref_gwf:
            plot( that.t, that.phi, linewidth=that_lwid, color=1-clr[0], alpha=that_alpha )
        # pylim( this.t, this.phi, domain=xlim )
        if tlim is not None:
            mask = (this.t>min(tlim)) & (this.t<max(tlim))
            ylim( lim( this.phi[mask], dilate=0.1 ) )
        yl( r'$\phi = \mathrm{arg}(%s)$' % kind.replace('$','') ,fontsize=fs,color=txclr, family=font_family)

        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        # Time domain frequency
        #--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
        sca( ax[2] )
        grid(color=gclr, linestyle='-')
        # Actual plotting
        plot( this.t, this.dphi, linewidth=lwid, color=sqrt(clr[0]) )
        if ref_gwf:
            plot( that.t, that.dphi, linewidth=that_lwid, color=sqrt(clr[0]), alpha=that_alpha )
        # pylim( this.t, this.dphi, domain=xlim )
        yl(r'$\mathrm{d}{\phi}/\mathrm{d}t$',fontsize=fs,color=txclr, family=font_family)

        if tlim is not None:
            mask = (this.t>min(tlim)) & (this.t<max(tlim))
            ylim( lim( this.dphi[mask], dilate=0.1 ) )

        # Full figure settings
        ax[0].set_xlim( lim(this.t) if tlim is None else tlim )
        if title is not None:
            ax[0].set_title( title, family=font_family )

        # Set axis lines (e.g. grid lines) below plot lines
        for a in ax:
            a.set_axisbelow(True)

        #
        xlabel(r'$t$',fontsize=fs,color=txclr)

        #
        if show:
            shw()

        #
        return ax

    # Apply a time domain window to the waveform. Either the window vector OR a set of indeces to be tapered is given as input. NOTE that while this method modifies the current object, one can revert to object's original state by using the reset() method. OR one can make a backup of the current object by using the clone() method.
    def apply_window( this,             # gwf object to be windowed
                      state = None,     # Index values defining region to be tapered:
                                        # For state=[a,b], if a>b then the taper is 1 at b and 0 at a
                                        # If a<b, then the taper is 1 at a and 0 at b.
                      window = None):   # optional input: use known taper/window

        # Store the initial state of the waveform array just in case the user wishes to undo the window
        this.__prevarr__ = this.wfarr

        # Use low level function
        if (state is not None) and (window is None):
            window = maketaper( this.t, state)
        elif (state is None) and (window is None):
            msg = '(!!) either "state" or "window" keyword arguments must be given and not None.'
            error(msg,'gwf.taper')

        #
        wfarr = this.wfarr
        wfarr[:,1] = window * this.wfarr[:,1]
        wfarr[:,2] = window * this.wfarr[:,2]

        # NOTE that objects cannot be redefined within their methods, but their properties can be changed. For this reason, the line below uses setfields() rather than gwf() to apply the taper.
        this.setfields( wfarr=wfarr )

        # Set this object's window
        this.window = this.window * window

    # Apply mask
    def apply_mask( this, mask=None ):
        #
        if mask is None: error('the mask input must be given, and it must be index or boolean ')
        #
        this.setfields( this.wfarr[mask,:] )

    # If desired, reset the waveform object to its original state (e.g. it's state just afer loading).
    # Note that after this methed is called, the current object will occupy a different address in memory.
    def reset(this): this.setfields( this.__rawgwfarr__ )

    # return a copy of the current object
    def copy(this):

        #
        from copy import deepcopy as copy
        return copy(this)

    # RETURN a clone the current waveform object. NOTE that the copy package may also be used here
    def clone(this): return gwf(this.wfarr).meet(this)

    # Interpolate the current object
    def interpolate(this,dt=None,domain=None):

        # Validate inputs
        if (dt is None) and (domain is None):
            msg = red('First "dt" or "domain" must be given. See traceback above.')
            error(msg,'gwf.interpolate')
        if (dt is not None) and (domain is not None):
            msg = red('Either "dt" or "domain" must be given, not both. See traceback above.')
            error(msg,'gwf.interpolate')

        # Create the new wfarr by interpolating
        if domain is None:
            wfarr = intrp_wfarr(this.wfarr,delta=dt)
        else:
            wfarr = intrp_wfarr(this.wfarr,domain=domain)

        # Set the current object to its new state
        this.setfields(wfarr)

    # Pad this waveform object in the time domain with zeros
    def pad(this,new_length=None,where=None,apply=False,extend=True):
        #
        where = 'right' if where is None else where
        # Pad this waveform object to the left and right with zeros
        ans = this.copy() if not apply else this
        if new_length is not None:
            # Create the new wfarr
            wfarr = pad_wfarr( this.wfarr, new_length,where=where,extend=extend )
            # Confer to the current object
            ans.setfields(wfarr)

        #
        if extend==False:
            if len(ans.t)!=new_length:
                error('!!!')

        return ans

    # Shift this waveform object in the time domain
    def tshift(this,shift=None,apply=False,method=None, verbose=False):

        # Pad this waveform object to the left and right with zeros
        ans = this.copy() if not apply else this
        if not (shift is None):
            # Create the new wfarr
            wfarr = tshift_wfarr( ans.wfarr, shift, method=method, verbose=verbose )
            # Confer to the current object
            ans.setfields(wfarr)

        return ans

    # Analog of the numpy ndarray conj()
    def conj(this):
        this.wfarr[:,2] *= -1
        this.setfields()
        return this

    # Align the gwf with a reference gwf using a desired method
    def align( this,
               that,            # The reference gwf object
               method=None,     # The alignment type e.g. phase
               options=None,    # Addtional options for subroutines
               mask=None,       # Boolean mask to apply for alignment (useful e.g. for average-phase alignment)
               kind=None,
               verbose=False ):

        #
        if that.__class__.__name__!='gwf':
            msg = 'first input must be gwf -- the gwf object to alignt the current object to'
            error(msg,'gwf.align')

        # Set default method
        if method is None:
            msg = 'No method chosen. We will proceed by aligning the waveform\'s initial phase.'
            warning(msg,'gwf.align')
            method = ['initial-phase']

        # Make sure method is list or tuple
        if not isinstance(method,(list,tuple)):
            method = [method]

        # Make sure all methods are strings
        for k in method:
            if not isinstance(k,str):
                msg = 'non-string method type found: %s'%k
                error(msg,'gwf.align')

        # Check for handled methods
        handled_methods = [ 'initial-phase','average-phase' ]
        for k in method:
            if not ( k in handled_methods ):
                msg = 'non-handled method input: %s. Handled methods include %s'%(red(k),handled_methods)
                error(msg,'gwf.align')

        #
        if kind is None: kind = 'srtain'

        # Look for phase-alignement
        if 'initial-phase' in method:
            this.wfarr = align_wfarr_initial_phase( this.wfarr, that.wfarr )
            this.setfields()
        if 'average-phase' in method:
            this.wfarr = align_wfarr_average_phase( this.wfarr, that.wfarr, mask=mask, verbose=verbose)
            this.setfields()

        #
        return this

    # Shift the waveform phase
    def shift_phase(this,
                    dphi,
                    fromraw=False,    # If True, rotate the wavefor relative to its default wfarr (i.e. __rawgwfarr__)
                    apply = True,
                    fast = False,
                    verbose=False):

        #
        from numpy import ndarray
        if isinstance(dphi,(list,tuple,ndarray)):
            if len(dphi)==1:
                dphi = dphi[0]
            else:
                error( 'dphi found to be iterable of length greater than one. the method is not implemented to handle this scenario. Please loop over desired values externally.' )
        if not isinstance(dphi,(float,int)):
            error('input must of float or int real valued','gwf.shift_phase')

        if fromraw:
            wfarr = this.__rawgwfarr__
        else:
            wfarr = this.wfarr

        #
        msg = 'This function could be sped up by manually aligning relevant fields, rather than regenerating all fields which includes taking an FFT.'
        if this.verbose: warning(msg,'gwf.shift_phase')

        #
        ans = this if apply else this.copy()
        wfarr = shift_wfarr_phase( wfarr, dphi )
        if fast:
            ans.setfields(wfarr,setfd=False)
        else:
            ans.setfields(wfarr)
        #
        if not apply:
            return ans

    #
    def __rotate_frame_at_all_times__( this,                        # The current object
                                       like_l_multipoles,           # List of available multipoles with same l
                                       euler_alpha_beta_gamma,      # List of euler angles
                                       ref_orientation = None ):    # A reference orienation (useful for BAM)

        #
        that = this.copy()

        #
        if not ( ref_orientation is None ) :
            error('The use of "ref_orientation" has been depreciated for this function.')

        #
        like_l_multipoles_dict = { (y.l,y.m):y.wfarr for y in like_l_multipoles }

        #
        rotated_wfarr = rotate_wfarrs_at_all_times( this.l,this.m, like_l_multipoles_dict, euler_alpha_beta_gamma, ref_orientation=ref_orientation )

        # Reset related fields using the new data
        that.setfields( rotated_wfarr )

        #
        return that

    # frequency domain filter the waveform given a window state for the frequency domain
    def fdfilter(this,window):
        #
        from scipy.fftpack import fft, fftfreq, fftshift, ifft
        from numpy import floor,array,log
        from matplotlib.pyplot import plot,show
        #
        if this.__lowpassfiltered__:
            msg = 'wavform already low pass filtered'
            warning(msg,'gwf.lowpass')
        else:
            #
            fd_y = this.fd_y * window
            plot( log(this.f), log( abs(this.fd_y) ) )
            plot( log(this.f), log( abs(fd_y) ) )
            show()
            #
            y = ifft( fftshift( fd_y ) )
            this.wfarr[:,1],this.wfarr[:,2] = y.real,y.imag
            #
            this.setfields()
            #
            this.__lowpassfiltered__ = True


# Class for waveforms: Psi4 multipoles, strain multipoles (both spin weight -2), recomposed waveforms containing h+ and hx. NOTE that detector response waveforms will be left to pycbc to handle
class gwylm:
    '''
    Class to hold spherical multipoles of gravitaiton wave radiation from NR simulations. A simulation catalog entry obejct (of the scentry class) as well as the l and m eigenvalue for the desired multipole (aka mode) is needed.
    '''

    # Class constructor
    def __init__( this,                             # reference for the object to be created
                  scentry_obj,                      # member of the scentry class
                  lm                    = None,     # iterable of length 2 containing multipolr l and m
                  lmax                  = None,     # if set, multipoles with all |m| up to lmax will be loaded.
                                                    # This input is not compatible with the lm tag
                  dt                    = 0.15,     # if given, the waveform array will beinterpolated to
                                                    # this timestep
                  load                  = None,     # IF true, we will try to load data from the scentry_object
                  clean                 = None,     # Toggle automatic tapering
                  extraction_parameter  = None,     # Extraction parameter labeling extraction zone/radius for run
                  level = None,                     # Opional refinement level for simulation. NOTE that not all NR groups use this specifier. In such cases, this input has no effect on loading.
                  w22 = None,                       # Optional input for lowest physical frequency in waveform; by default an wstart value is calculated from the waveform itself and used in place of w22
                  lowpass=None,                     # Toggle to lowpass filter waveform data upon load using "romline" (in basics.py) routine to define window
                  calcstrain = None,                # If True, strain will be calculated upon loading
                  calcnews = None,
                  enforce_polarization_convention = None, # If true, polarization will be adjusted according to initial separation vectors
                  fftfactor = None,                   # Option for padding wfarr to next fftfactor powers of two
                  pad = None,                       # Optional padding length in samples of wfarr upon loading; not used if fftfactor is present; 'pad' samples dwill be added to the wfarr rows
                  __M_RELATIVE_SIGN_CONVENTION__ = None,
                  initial_j_align = None,           # Toggle for putting wabeform in frame where initial J is z-hat
                  verbose               = None ):   # be verbose

        '''
        Put something informative here!
        '''

        # NOTE that this method is setup to print the value of each input if verbose is true.
        # NOTE that default input values are handled just below

        # Print non None inputs to screen
        thisfun = this.__class__.__name__
        if not ( verbose in (None,False) ):
            for k in dir():
                if (eval(k) is not None) and (eval(k) is not False) and not ('this' in k):
                    msg = 'Found %s (=%r) keyword.' % (textul(k),eval(k))
                    alert( msg, 'gwylm' )

        # Handle default values
        load = True if load is None else load
        clean = False if clean is None else clean
        calcstrain = True if calcstrain is None else calcstrain
        calcnews = True if calcnews is None else calcnews
        this.enforce_polarization_convention = False if enforce_polarization_convention is None else enforce_polarization_convention

        # Validate the lm input
        this.__valinputs__(thisfun,lm=lm,lmax=lmax,scentry_obj=scentry_obj)

        # Allow users to give directory instead of scentry object becuase it's easier for some people
        if isinstance(scentry_obj,str):
            simdir = scentry_obj
            warning( 'You have input a directory rather than an scentry object. We will try to convert the directory to an scentry object, but this is slower than using the our catalog system. Please consider modifying the appropriate configuretion file (i.e. in "%s") to accommodate your new simulation, or perhaps create a new configuration file. Given your new or updated configuration file, please run nrutils.scbuild("my_config_name") to update your local catalog. If you are confident that all has gone well, you may also wish to push changes in your catalog (to the master repo). Live long and prosper. -- Lionel'%cyan(global_settings.config_path),'gwylm' )
            scentry_obj = simdir2scentry( simdir, verbose=verbose )[0]

        # Allow users to input path to h5 file in lvc-nr format

        # Confer the scentry_object's attributes to this object for ease of referencing
        for attr in scentry_obj.__dict__.keys():
            setattr( this, attr, scentry_obj.__dict__[attr] )

        # NOTE that we don't want the scentry's verbose property to overwrite the input above, so we definte this.verbose at this point, not before.
        this.verbose = verbose

        #
        this.frame = 'initial-simulation'

        # #
        # if fftfactor is None:
        #     warning('No fftfactor input given. As a matter of caution, we will set it to 1, meaning that the data length will be extended symmetrically to the next power of 2.')
        #     fftfactor = 1

        #
        this.fftfactor = fftfactor

        #
        # from numpy import sign
        # warning('OVERRIDING M_RELATIVE_SIGN_CONVENTION using standard based on z-compoenent of total spin')
        # __M_RELATIVE_SIGN_CONVENTION__ = sign(this.S[-1])
        if this.verbose:
            if __M_RELATIVE_SIGN_CONVENTION__ is None:
                alert('Using default M_RELATIVE_SIGN_CONVENTION of %i'%M_RELATIVE_SIGN_CONVENTION)
            else:
                alert('Using %s M_RELATIVE_SIGN_CONVENTION of %i'%(yellow('input'),__M_RELATIVE_SIGN_CONVENTION__))
        this.M_RELATIVE_SIGN_CONVENTION = M_RELATIVE_SIGN_CONVENTION if __M_RELATIVE_SIGN_CONVENTION__ is None else __M_RELATIVE_SIGN_CONVENTION__

        # Store the scentry object to optionally access its methods
        this.__scentry__ = scentry_obj

        ''' Explicitely reconfigure the scentry object for the current user. '''
        # this.config.reconfig() # NOTE that this line is commented out because scentry_obj.simdir() below calls the reconfigure function internally.

        # Tag this object with the simulation location of the given scentry_obj. NOTE that the right hand side of this assignment depends on the user's configuration file. Also NOTE that the configuration object is reconfigured to the system's settings within simdir()
        this.simdir = scentry_obj.simdir()

        # If no extraction parameter is given, retrieve default. NOTE that this depends on the current user's configuration.
        # NOTE that the line below is commented out becuase the line above (i.e. ... simdir() ) has already reconfigured the config object
        # scentry_obj.config.reconfig() # This line ensures that values from the user's config are taken
        if extraction_parameter is None:
            extraction_parameter = scentry_obj.default_extraction_par
        if level is None:
            level = scentry_obj.default_level

        # There will be a common time for all multipoles. This time will be set either upon loading of psi4 or upon calculation of strain
        this.t = None

        #
        if scentry_obj.config:
            config_extraction_parameter = scentry_obj.config.default_par_list[0]
            config_level = scentry_obj.config.default_par_list[1]
            if (config_extraction_parameter,config_level) != (extraction_parameter,level):
                msg = 'The (%s,%s) is (%s,%s), which differs from the config values of (%s,%s). You have either manually input the non-config values, or the handler has set them by looking at the contents of the simulation directory. '%(magenta('extraction_parameter'),green('level'),magenta(str(extraction_parameter)),green(str(level)),str(config_extraction_parameter),str(config_level))
                if this.verbose: alert( msg, 'gwylm' )

        # Store the extraction parameter and level
        this.extraction_parameter = extraction_parameter
        # print this.extraction_map_dict['level_map']
        if this.extraction_map_dict['level_map']:
            this.level = this.extraction_map_dict['level_map'][extraction_parameter]
        else:
            this.level = this.extraction_parameter

        # Store the extraction radius if a map is provided in the handler file
        if 'loadhandler' in scentry_obj.__dict__:
            special_method,handler = 'extraction_map',scentry_obj.loadhandler()
            if special_method in handler.__dict__:
                this.extraction_radius = handler.__dict__[special_method]( scentry_obj, this.extraction_parameter )
            else:
                this.extraction_radius = None

        # These fields are initiated here for visiility, but they are filled as lists of gwf object in load()
        this.ylm,this.hlm,this.flm = [],[],[] # psi4 (loaded), strain(calculated by default), news(optional non-default)

        # time step
        this.dt = dt

        # Load the waveform data
        if load==True:
            this.__load__(lmax=lmax,lm=lm,dt=dt,pad=pad)

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # Enforce polarization convention based on intial compoenent positions
        if this.enforce_polarization_convention:
            from numpy import arctan2,sin,cos,dot
            R = -this.R2+this.R1
            dpsi_initial = arctan2( R[1], R[0] )
            this.rotate( dpsi=dpsi_initial, verbose=False )
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        # Characterize the waveform's start and store related information to this.preinspiral
        this.preinspiral = None # In charasterize_start(), the information about the start of the waveform is actually stored to "starting". Here this field is inintialized for visibility.
        if scentry_obj.config:
            # Only do this if the scentry object has config information; otheriwse, it was probably created outside of scbuild and will probably break code
            this.characterize_start_end()

        # If w22 is input, then use the input value for strain calculation. Otherwise, use the algorithmic estimate.
        if scentry_obj.config:
            if w22 is None:
                w22 = this.wstart_pn
                if verbose:
                    # msg = 'Using w22 from '+bold(magenta('algorithmic estimate'))+' to calculate strain multipoles.'
                    msg = 'Storing w22 from a '+bold(magenta('PN estimate'))+'[see pnw0 in basics.py, and/or arxiv:1310.1528v4]. This will be the frequency parameter used if strain is to be calculated.'
                    alert( msg, 'gwylm' )
            else:
                if verbose:
                    msg = 'Storing w22 from '+bold(magenta('user input'))+'.  This will be the frequency parameter used if strain is to be calculated.'
                    alert( msg, 'gwylm' )

        # Low-pass filter waveform (Psi4) data using "romline" routine in basics.py to determin windowed region
        this.__lowpassfiltered__ = False
        if lowpass:
            this.lowpass()

        # Calculate news
        if calcnews and scentry_obj.config:
            this.calcflm(w22=w22)

        # Calculate strain
        if calcstrain and scentry_obj.config:
            this.calchlm(w22=w22)

        # Clean the waveforms of junk radiation if desired
        this.__isclean__ = False
        if clean:
            this.clean()

        # Set some boolean tags
        this.__isringdownonly__ = False # will switch to True if, ringdown is cropped. See gwylm.ringdown().

        # Create a dictionary representation of the mutlipoles
        if scentry_obj.config:
            this.__curate__()

        #
        this.__enforce_m_relative_phase_orientation__()


    # Allow class to be indexed
    def __getitem__(this,index):
        # Import usefuls
        from numpy import ndarray
        # Define validation message
        msg = 'Note that gwylm objects can only be indexed with (l,m) pairs corresponding to the data loaded upon its creation. The attempted index is %s.'%(str(index))
        #
        if isinstance(index,int): index = this.__lmlist__[index]
        # Return the l,m dictionary
        if isinstance(index,(tuple,list,ndarray)):
            if len(index) != 2:
                error(msg+' Length of reference must be 2; one for spherical harmonic l another for m.')
        else:
            error(msg)
        return this.lm[index]
    # Allow "in" to look for l,m content
    def __contains__(this,query):
        print query
        return query in this.__lmlist__
    # Allow class to be iterable as its __lmlist__
    def __iter__(this):
        return this.__lmlist__
    # def next(this):
    #     k = this.__lmlist__.index(this.__iterLM__)
    #     N = len(this.__lmlist__)
    #     print k
    #     if k > (N-1):
    #         raise StopIteration
    #     else:
    #         return this.__lmlist__[k+1]

    #
    def get_radiation_axis_info(this,kind='psi4',plot=False,save=False):
        #
        from nrutils.manipulate.rotate import gwylm_radiation_axis_workflow
        return gwylm_radiation_axis_workflow(this,kind=kind,plot=plot,save=save,verbose=this.verbose)

    # Create a dictionary representation of the mutlipoles
    def __curate__(this):
        '''Create a dictionary representation of the mutlipoles'''
        # NOTE that this method should be called every time psi4, strain and/or news is loaded.
        # NOTE that the related methods are: __load__, calchlm and calcflm
        # Initiate the dictionary
        this.lm = {}
        for l,m in this.__lmlist__:
            this.lm[l,m] = {}
        # Seed the dictionary with psi4 gwf objects
        for y in this.ylm:
            this.lm[(y.l,y.m)]['psi4'] = y
        # Seed the dictionary with strain gwf objects
        for h in this.hlm:
            this.lm[(h.l,h.m)]['strain'] = h
        # Seed the dictionary with strain gwf objects
        for f in this.flm:
            this.lm[(f.l,f.m)]['news'] = f
        #
        this.t = this[2,2]['psi4'].t
        this.f = this[2,2]['psi4'].f

    # Validate inputs to constructor
    def __valinputs__(this,thisfun,lm=None,lmax=None,scentry_obj=None):

        from numpy import shape

        # Raise error upon nonsensical multipolar input
        if (lm is not None) and (lmax is not None) and load:
            msg = 'lm input is mutually exclusive with the lmax input'
            raise NameError(msg)

        # Default multipolar values
        if (lm is None) and (lmax is None):
            lm = [2,2]

        # Determine whether the lm input is a songle mode (e.g. [2,2]) or a list of modes (e.g. [[2,2],[3,3]] )
        if len( shape(lm) ) == 2 :
            if shape(lm)[1] != 2 :
                # raise error
                msg = '"lm" input must be iterable of length 2 (e.g. lm=[2,2]), or iterable of shape (X,2) (e.g. [[2,2],[3,3],[4,4]])'
                error(msg,thisfun)

        # Raise error upon nonsensical multipolar input
        if not isinstance(lmax,int) and lm is None:
            msg = '(!!) lmax must be non-float integer.'
            raise ValueError(msg)

        # Make sure that only one scentry in instput (could be updated later)
        if not ((scentry_obj.__class__.__name__ == 'scentry') or isinstance(scentry_obj,str)):
            msg = 'First input must be member of scentry class (e.g. as returned from scsearch() ). OR it must be a path where a valid metadata file exists. See simdir2scentry() for more info.'
            error(msg,thisfun)

    # Make a list of lm values related to this gwylm object
    def __make_lmlist__( this, lm, lmax ):

        #
        from numpy import shape

        #
        this.__lmlist__ = []

        # If if an lmax value is given.
        if lmax is not None:
            # Then load all multipoles within lmax
            for l in range(2,lmax+1):
                #
                for m in range(-l,l+1):
                    #
                    this.__lmlist__.append( (l,m) )
        else: # Else, load the given lis of lm values
            # If lm is a list of specific multipole indeces
            if isinstance(lm[0],(list,tuple)):
                #
                for k in lm:
                    if len(k)==2:
                        l,m = k
                        this.__lmlist__.append( (l,m) )
                    else:
                        msg = '(__make_lmlist__) Found list of multipole indeces (e.g. [[2,2],[3,3]]), but length of one of the index values is not two. Please check your lm input.'
                        error(msg)
            else: # Else, if lm is a single mode index
                #
                l,m = lm
                this.__lmlist__.append( (l,m) )

        # Store the input lm list
        this.__input_lmlist__ = list(this.__lmlist__)
        # Always load the m=l=2 waveform
        if not (  (2,2) in this.__lmlist__  ):
            msg = 'The l=m=2 multipole will be loaded in order to determine important characteristice of all modes such as noise floor and junk radiation location.'
            warning(msg,'gwylm')
            this.__lmlist__.append( (2,2) )

        # Always put (2,2) at the front of the list
        a = list(this.__lmlist__)
        a.pop( a.index((2,2)) )
        this.__lmlist__ = [(2,2)] + a

        # Let the people know
        if this.verbose:
            alert('The following spherical multipoles will be loaded:%s'%cyan(str(this.__lmlist__)))

    # Wrapper for core load function. NOTE that the extraction parameter input is independent of the usage in the class constructor.
    def __load__( this,                      # The current object
                  lmax=None,                 # max l to use
                  lm=None,                   # (l,m) pair or list of pairs to use
                  extraction_parameter=None, # the label for different extraction zones/radii
                  level = None,              # Simulation resolution level (Optional and not supported for all groups )
                  dt=None,
                  pad = None, # optional padding length for wfarr upon load
                  verbose=None ):

        #
        from numpy import shape

        # Make a list of l,m values and store it to the current object as __lmlist__
        this.__make_lmlist__( lm, lmax )

        # Load all values in __lmlist__
        this.external_sign_convention = None
        for lm in this.__lmlist__:
            this.load(lm=lm,dt=dt,extraction_parameter=extraction_parameter,level=level,pad=pad,verbose=verbose)

        # Ensuer that all modes are the same length
        this.__valpsi4multipoles__()

        # Create a dictionary representation of the mutlipoles
        this.__curate__()

    # Validate individual multipole against the l=m=2 multipole: e.g. test lengths are same
    def __valpsi4multipoles__(this):
        #
        this.__curate__()
        #
        t22 = this.lm[2,2]['psi4'].t
        n22 = len(t22)
        #
        for lm in this.lm:
            if lm != (2,2):
                ylm = this.lm[lm]['psi4']
                if len(ylm.t) != n22:
                    #
                    if True: #this.verbose:
                        warning('[valpsi4multipoles] The (l,m)=(%i,%i) multipole was found to not have the same length as its (2,2) counterpart. The offending waveform will be interpolated on the l=m=2 time series.'%lm,'gwylm')
                    # Interpolate the mode at t22, and reset fields
                    wfarr = intrp_wfarr(ylm.wfarr,domain=t22)
                    # Reset the fields
                    ylm.setfields(wfarr=wfarr)


    #Given an extraction parameter, use the handler's extraction_map to determine extraction radius
    def __r__(this,extraction_parameter):
        #
        return this.__scentry__.loadhandler().extraction_map(this,extraction_parameter)

    # load the waveform data
    def load(this,                  # The current object
             lm=None,               # the l amd m values of the multipole to load
             file_location=None,    # (Optional) is give, this file string will be used to load the file,
                                    # otherwise the function determines teh file string automatically.
             dt = None,             # Time step to enforce for data
             extraction_parameter=None,
             level=None,            # (Optional) Level specifyer for simulation. Not all simulation groups use this!
             output=False,          # Toggle whether to store data to the current object, or output it
             pad = None, # optional padding length for wfarr upon load
             verbose=None):

        # Import useful things
        from os.path import isfile,basename
        from numpy import sign,diff,unwrap,angle,amax,isnan,amin,log,exp,std,median,mod,mean
        from scipy.stats.mstats import mode
        from scipy.version import version as scipy_version
        thisfun=inspect.stack()[0][3]

        # Handle the verbose input. NOTE that this toggle is possibly not used. See "this.verbose".
        if verbose is None: verbose = this.verbose

        # Default multipolar values
        if lm is None:
            lm = [2,2]

        # Raise error upon nonsensical multipolar input
        if lm is not None:
            if len(lm) != 2 :
                msg = '(!!) lm input must contain iterable of length two containing multipolar indeces'
                raise ValueError(msg)
        if abs(lm[1]) > lm[0]:
            msg = '(!!) Note that m=lm[1], and it must be maintained that abs(m) <= lm[0]=l. Instead (l,m)=(%i,%i).' % (lm[0],lm[1])
            raise ValueError(msg)
        # If file_location is not string, then let the people know.
        if not isinstance( file_location, (str,type(None)) ):
            msg = '(!!) '+yellow('Error. ')+'Input file location is type %s, but must instead be '+green('str')+'.' % magenta(type(file_location).__name__)
            raise ValueError(msg)

        # NOTE that l,m and extraction_parameter MUST be defined for the correct file location string to be created.
        l = lm[0]; m = lm[1]

        # Load default file name parameters: extraction_parameter,l,m,level
        if extraction_parameter is None:
            # Use the default value
            extraction_parameter = this.extraction_parameter
            if verbose: alert('Using the '+cyan('default')+' extraction_parameter of %g' % extraction_parameter)
        else:
            # Use the input value
            this.extraction_parameter = extraction_parameter
            if verbose: alert('Using the '+cyan('input')+' extraction_parameter of '+cyan('%g' % extraction_parameter))
        if level is None:
            # Use the default value
            level = this.level
            if verbose: alert('Using the '+cyan('default')+' level of %g' % level)
        else:
            # Use the input value
            this.level = level
            if verbose: alert('Using the '+cyan('input')+' level of '+cyan('%g' % level))

        # This boolean will be set to true if the file location to load is found to exist
        proceed = False

        # Construct the string location of the waveform data. NOTE that config is inhereted indirectly from the scentry_obj. See notes in the constructor.
        if file_location is None: # Find file_location automatically. Else, it must be input

            # file_location = this.config.make_datafilename( extraction_parameter, l,m )

            # For all formatting possibilities in the configuration file
             # NOTE standard parameter order for every simulation catalog
             # extraction_parameter l m level
            for fmt in this.config.data_file_name_format :

                # NOTE the ordering here, and that the filename format in the config file has to be consistent with: extraction_parameter, l, m, level
                file_location = (this.simdir + fmt).format( extraction_parameter, l, m, level )
                # OLD Formatting Style:
                # file_location = this.simdir + fmt % ( extraction_parameter, l, m, level )

                # test whether the file exists
                if isfile( file_location ):
                    break

        # If the file location exists, then proceed. If not, then this error is handled below.
        if isfile( file_location ):
            proceed = True

        # If the file to be loaded exists, then load it. Otherwise raise error.
        if proceed:

            # load array data from file
            if this.verbose: alert('Loading: %s' % cyan(basename(file_location)) )
            wfarr,_ = smart_load( file_location, verbose=this.verbose )

            # Handle extraction radius scaling
            if not this.config.is_rscaled:
                # If the data is not in the format r*Psi4, then multiply by r (units M) to make it so
                extraction_radius = this.__r__(extraction_parameter)
                wfarr[:,1:3] *= extraction_radius

            # Fix nans, nonmonotinicities and jumps in time series waveform array
            # NOTE that the line below is applied within the gwf constructor
            # wfarr = straighten_wfarr( wfarr )

            # Make sure that waveform array is straight
            wfarr = straighten_wfarr(wfarr,this.verbose)
            # Make sure that it's equispaced
            if (std(diff(wfarr[:,0]))>1e-6): dt = mean(diff(wfarr[:,0])) if dt is None else dt
            if (dt is not None) or (std(diff(wfarr[:,0]))>1e-6):
                wfarr = intrp_wfarr( wfarr, dt, verbose = not True )
            # NOTE: If no specific padding is requested, we will still pad the data by some small amount to enforce that the start and end values are identical prior to propper cleaning. This results in noticeable advatances in data quality when computing matches with short waveforms.
            if not isinstance(pad,(int,float)):
                old_data_length = len(wfarr[:,0])
                default_pad = 2 + mod(len(wfarr[:,0]),2) + 1
                if this.verbose: alert('Imposing a default padding of %i to the data.'%default_pad)
                # NOTE that the optional "where" input below has important implications for many algorithms used for processing. In short, padding to the right only ensures that the original "start" of the waveform data series is unchanged.
                wfarr = pad_wfarr(wfarr,default_pad,where='right',verbose=this.verbose)

            # Pad the waveform array
            if (this.fftfactor != 0) and (this.fftfactor is not None):
                # error('the fftfactor option seems to give strange results for td strain, and is thus currently disabled --- THIS WAS FIXED by calling straighten_wfarr within pad_wfarr.')
                warning('Enabling the fftfactor option can sometimes cause strange behavior. Use with caution. Make sure that time and frequency domain waveforms are as expected.')
                if isinstance(this.fftfactor,int):
                    # pad the waveform array in the time domain
                    # NOTE that this is preferable to simply using the "n" input in fft calls
                    # becuase we wish the time and frequency domain data to be one-to-one under ffts
                    old_data_length = len(wfarr[:,0])
                    fftlen = int( 2 ** ( int(log( old_data_length )/log(2)) + 1.0 + this.fftfactor ) )
                    #
                    if this.verbose: alert( 'Padding wfarr. The old data length was %i, and the new one is %i'%(old_data_length,fftlen) )
                    # NOTE that this padding function only works with time domain data
                    wfarr = pad_wfarr(wfarr,fftlen,where='sides',verbose=this.verbose)
                else:
                    error('fftfactor must be int corresponding to additional powers of 2 to which the data will be padded symetrically')
            else:
                #
                if isinstance(pad,(int,float)):
                    # warning('Enabling the pad option can sometimes cause strange behavior. Use with caution. Make sure that time and frequency domain waveforms are as expected.')
                    pad = int(pad)
                    # pad the waveform array in the time domain
                    # NOTE that this is preferable to simply using the "n" input in fft calls
                    # becuase we wish the time and frequency domain data to be one-to-one under ffts
                    old_data_length = len(wfarr[:,0])
                    new_data_length = old_data_length + pad
                    #
                    if this.verbose: alert( 'Padding wfarr. The old data length was %i, and the new one is %i'%(old_data_length,new_data_length) )
                    # NOTE that this padding function only works with time domain data
                    wfarr = pad_wfarr(wfarr,new_data_length,where='sides',verbose=this.verbose)

            # Initiate waveform object and check that sign convetion is in accordance with core settings
            def mkgwf(wfarr_):
                return gwf( wfarr_,
                            l=l,
                            m=m,
                            extraction_parameter=extraction_parameter,
                            dt=dt,
                            verbose=this.verbose,
                            mf = this.mf,
                            m1 = this.m1, m2 = this.m2,
                            xf = this.xf,
                            label = this.label,
                            ref_scentry = this.__scentry__,
                            kind='$rM\psi_{%i%i}$'%(l,m) )

            #
            y_ = mkgwf(wfarr)

            # # ---------------------------------------------------- #
            # # Enforce internal sign convention for Psi4 multipoles
            # '''
            # NOTE that the change enforced here is equivalent to changing
            # the convention used when defining the time domain phase: e^i*phi
            # or e^-i*phi. NOTE that this convention is independent of the
            # symmetries of the radiation (e.g. Ylm = -1^l * conj(Yl,-m) ).
            # '''
            # # ---------------------------------------------------- #

            # # Try to determine the sign convention used to define phase. Note that this will be determined only once for the current object based on the l=m=2 multipole.
            # if this.external_sign_convention is None:
            #     msk_ = y_.amp > 0.0001*amax(y_.amp)
            #     # msk_ = y_.amp > 0.01*amax(y_.amp)
            #     if int(scipy_version.split('.')[1])<16:
            #         # Account for old scipy functionality
            #         external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( y_.dphi[msk_] ) )[0][0]
            #         initially_msign_matches_wsign = sign(m) == mode( sign( y_.dphi[msk_] ) )[0][0]
            #     else:
            #         # Account for modern scipy functionality
            #         external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( y_.dphi[msk_] ) ).mode[0]
            #         initially_msign_matches_wsign = sign(m) == mode( sign( y_.dphi[msk_] ) ).mode[0]
            #     if initially_msign_matches_wsign: alert('## initall, m and td freq have same sign.')
            #     this.external_sign_convention = external_sign_convention
            #
            # if this.M_RELATIVE_SIGN_CONVENTION != this.external_sign_convention:
            #     wfarr[:,2] = -wfarr[:,2]
            #     y_ = mkgwf(wfarr)
            #     # Let the people know what is happening.
            #     msg = yellow('Re-orienting waveform phase')+' to be consistent with internal sign convention for Psi4, where sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]).' % this.M_RELATIVE_SIGN_CONVENTION + ' Note that the internal sign convention is defined in ... nrutils/core/__init__.py as "M_RELATIVE_SIGN_CONVENTION". This message has appeared becuase the waveform is determioned to obey and sign convention: sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]). Note the appearance of the initial z angular momentum, this.L[-1].'%(this.external_sign_convention)
            #     thisfun=inspect.stack()[0][3]
            #     warning( msg, verbose=this.verbose )

            # use array data to construct gwf object with multipolar fields
            if not output:
                this.ylm.append( y_ )
                if this.t is None: this.t = y_.t
            else:
                return y_

        else:

            # There has been an error. Let the people know.
            msg = '(!!) Cannot find "%s". Please check that catalog_dir and data_file_name_format in %s are as desired. Also be sure that input l and m are within ranges that are actually present on disk.' % ( red(file_location), magenta(this.config.config_file_location) )
            raise NameError(msg)

    # Plotting function for class: plot plus cross amp phi of waveforms USING the plot function of gwf()
    def plot(this,show=False,fig=None,kind=None,verbose=False,domain=None,tlim=None,flim=None):
        #
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import figure
        from numpy import array,diff,pi

        # Handle default kind of waveform to plot
        if kind is None:
            kind = 'both'

        # Handle which default domain to plot
        if domain is None:
            domain = 'time'
        elif not ( domain in ['time','freq'] ):
            msg = 'Error: domain keyword must be either "%s" or "%s".' % (cyan('time'),cyan('freq'))
            error(msg,'gwylm.plot')

        # If the plotting of only psi4 or only strain is desired.
        if kind != 'both':

            # Handle kind options
            if kind in ['psi4','y4','psilm','ylm','psi4lm','y4lm']:
                wflm = this.ylm
            elif kind in ['hlm','h','strain']:
                # Determine whether to calc strain here. If so, then let the people know.
                if len(this.hlm) == 0:
                    msg = '(**) You have requested that strain be plotted before having explicitelly called gwylm.calchlm(). I will now call calchlm() for you.'
                    print magenta(msg)
                    this.calchlm()
                # Assign strain to the general placeholder.
                wflm = this.hlm

            # Plot waveform data
            for y in wflm:

                #
                fig = figure( figsize = 1.1*array([8,7.2]) )
                fig.set_facecolor("white")

                ax,_ = y.plot(fig=fig,title='%s: %s, (l,m)=(%i,%i)' % (this.setname,this.label,y.l,y.m),domain=domain,tlim=tlim,flim=flim)
                # If there is start characterization, plot some of it
                if 'starting' in this.__dict__:
                    clr = 0.4*array([1./0.6,1./0.6,1])
                    dy = 100*diff( ax[0].get_ylim() )
                    for a in ax:
                        dy = 100*diff( a.get_ylim() )
                        if domain == 'time':
                            a.plot( wflm[0].t[this.startindex]*array([1,1]) , [-dy,dy], ':', color=clr )
                        if domain == 'freq':
                            a.plot( this.wstart*array([1,1])/(2*pi) , [-dy,dy], ':', color=clr )
            #
            if show:
                # Let the people know what is being plotted.
                if verbose: print cyan('>>')+' Plotting '+darkcyan('%s'%kind)
                shw()

        else: # Else, if both are desired

            # Plot both psi4 and strain
            for kind in ['psi4lm','hlm']:
                ax = this.plot(show=show,kind=kind,domain=domain,tlim=tlim,flim=flim)

        #
        return ax


    # Make sky-plot for waveform at fixed time
    def mollweide_plot(this,time=None,kind=None,ax=None,form=None,colorbar_shrink=1.0,N=120,use_time_relative_to_peak = True):

        '''
        Make mollweide plot for waveform at a specified time instance relative to peak.
        See positive.plotting.sYlm_mollweide_plot for reference.
        '''

        # Handle optionals
        if kind is None: kind = 'strain'
        if time is None: time = 0

        # Import usefuls
        from matplotlib.pyplot import subplots,gca,gcf,figure,colorbar,draw
        from numpy import array,pi,linspace,meshgrid,zeros

        # Coordinate arrays for the graphical representation
        x = linspace(-pi, pi, N)
        y = linspace(-pi/2, pi/2, N/2)
        X,Y = meshgrid(x,y)

        # Spherical coordinate arrays derived from x, y
        theta = pi/2 - y
        phi = x.copy()

        # Determine the peak realtive time series
        if kind in ('h','strain'):
            kind = 'strain'
            peak_relative_time = this.lm[2,2]['strain'].t - this.lm[2,2]['strain'].intrp_t_amp_max
        elif kind in ('y','psi4'):
            kind = 'psi4'
            peak_relative_time = this.lm[2,2]['psi4'].t - this.lm[2,2]['psi4'].intrp_t_amp_max
        else:
            error('kind can only be "strain" or "psi4", not %s'%(str(kind)))
        # Define the mask which selects for time
        # NOTE that this is an index value, not a time value
        if use_time_relative_to_peak:
            k = find( peak_relative_time >= time )[0]
        else:
            k = find( this.t >= time )[0]
        # Store actual time referened for waveform
        real_time = this.lm[2,2][kind].t[k]

        # Recompose the waveform at this time evaluated over the source's sky
        Z = zeros( X.shape, dtype=complex )
        for l,m in this.lm:
            Z += this.lm[l,m][kind].y[k] * sYlm(-2,l,m,theta,phi).T

        #
        if form in (None,'+','plus'):
            Z = Z.real
            title = r'$\Re \left[ %s(t) \right]$'%( 'h' if kind == 'strain' else r'\psi_4' )
        elif form in ('x','cross'):
            Z = Z.imag
            title = r'$\Im \left[ %s(t) \right]$'%( 'h' if kind == 'strain' else r'\psi_4' )
        elif form in ('a','abs'):
            Z = abs(Z)
            title = r'$|%s(t)|$'%( 'h' if kind == 'strain' else r'\psi_4' )

        #
        title += ', $t = %1.4f$'%this.lm[2,2][kind].t[k]

        xlabels = ['$210^\circ$', '$240^\circ$','$270^\circ$','$300^\circ$','$330^\circ$',
                   '$0^\circ$', '$30^\circ$', '$60^\circ$', '$90^\circ$','$120^\circ$', '$150^\circ$']

        ylabels = ['$165^\circ$', '$150^\circ$', '$135^\circ$', '$120^\circ$',
                   '$105^\circ$', '$90^\circ$', '$75^\circ$', '$60^\circ$',
                   '$45^\circ$','$30^\circ$','$15^\circ$']

        #
        if ax is None:
            fig, ax = subplots(subplot_kw=dict(projection='mollweide'), figsize= 1*array([10,8]) )

        #
        im = ax.pcolormesh(X,Y,Z)
        ax.set_xticklabels(xlabels, fontsize=14)
        ax.set_yticklabels(ylabels, fontsize=14)
        ax.set_xlabel(r'$\phi$', fontsize=20)
        ax.set_ylabel(r'$\theta$', fontsize=20)
        ax.grid()
        colorbar(im, ax=ax, orientation='horizontal',shrink=colorbar_shrink,label=title)
        gcf().canvas.draw_idle()

        #
        return ax,real_time


    # Save multipoles to ascii files
    def saveto( this, outdir=None, kind = None, verbose = None ):

        # Import usefuls
        from os.path import expanduser,isfile,isdir,join
        from numpy import zeros,savetxt

        # Handle optional inputs
        if verbose is None: verbose = this.verbose
        if kind is None: kind = 'psi4'
        # Validate kind of data to save
        if not ( kind in this.lm[2,2] ):
            error('Unknown "kind" given: %s. Must be in %s.'%(kind,this.lm[2,2].keys()))
        # Handle default output dir
        if outdir is None: outdir = expanduser('~/Desktop/')

        # Automatically add simulation name to output dir
        outdir = join( outdir, this.simname )

        # Check for existance of output directory, and make if needed
        mkdir( outdir, verbose=verbose )

        # For all multipole indeces
        for lm in this.lm:

            # Open file for writing
            filename = kind+'_%s_l%im%i.asc'%((this.__scentry__.config.institute,)+lm)
            ascii_file = join( outdir, filename )
            f = open( ascii_file,'w')
            f.write('# Written by nrutils.gwylm.saveto() ~ gotta heart koalas\n')
            f.write('# t \t\t Re(%s) \t\t Im(%s)\n'%(kind,kind))
            data_array = zeros( (len(this.t),3) )
            data_array[:,0] = this.t
            data_array[:,1] = this[lm][kind].plus
            data_array[:,2] = this[lm][kind].cross
            savetxt( f, data_array )
            # close the file
            alert('ascii data stored to "%s"'%cyan(ascii_file),verbose=verbose)
            f.close()




    # Strain via ffi method
    def calchlm(this,w22=None):

        # Calculate strain according to the fixed frequency method of http://arxiv.org/pdf/1006.1632v3

        #
        from numpy import array,double

        # If there is no w22 given, then use the internally defined value of wstart
        if w22 is None:
            # w22 = this.wstart
            # NOTE: here we choose to use the spin independent PN estimate as a lower bound for the l=m=2 mode.
            w22 = this.wstart_pn


        # Reset
        this.hlm = []
        for y in this.ylm:

            # Calculate the strain for each part of psi4. NOTE that there is currently NO special sign convention imposed beyond that used for psi4.
            w0 = w22 * double(y.m)/2.0 # NOTE that wstart is defined in characterize_start_end() using the l=m=2 Psi4 multipole.
            # Here, m=0 is a special case
            if 0==y.m: w0 = w22
            # Let the people know
            if this.verbose:
                alert( magenta('w%i%i = m*w22/2 = %f' % (y.l,y.m,w0) )+yellow(' (this is the lower frequency used for FFI method [arxiv:1006.1632v3])') )

            # Create the core waveform information
            t       =  y.t
            h_plus  =  ffintegrate( y.t, y.plus,  w0, 2 )
            h_cross =  ffintegrate( y.t, y.cross, w0, 2 )\

            ## NOTE that interpolative intregration has been tried below.
            ## This does not appear to correct for the low frequency drift, so
            ## the above fixed frequency aproach is kept.
            # alert('Using spline!')
            # h_plus  = spline_antidiff(t,y.plus, n=2)
            # h_cross = spline_antidiff(t,y.cross,n=2)


            #%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%#
            # NOTE that there is NOT a minus sign above which is INconsistent with equation 3.4 of
            # arxiv:0707.4654v3. Here we choose to be consistent with eq 4 of arxiv:1006.1632 and not add a
            # minus sign.
            if this.verbose:
                msg = yellow('The user should note that there is no minus sign used in front of the double time integral for strain (i.e. Eq 4 of arxiv:1006.1632). This differs from Eq 3.4 of arxiv:0707.4654v3. The net effect is a rotation of the overall polarization of pi degrees. The user should also note that there is no minus sign applied to h_cross meaning that the user must be mindful to write h_plus-1j*h_cross when appropriate.')
                alert(msg,'gwylm.calchlm')
            #%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%.%%#

            # Constrcut the waveform array for the new strain object
            wfarr = array( [ t, h_plus, h_cross ] ).T
            if this.t is None: this.t = t

            # Add the new strain multipole to this object's list of multipoles
            this.hlm.append( gwf( wfarr, l=y.l, m=y.m, mf=this.mf, xf=this.xf, kind='$rh_{%i%i}/M$'%(y.l,y.m)) )

        # Create a dictionary representation of the mutlipoles
        this.__curate__()

        # NOTE that this is the end of the calchlm method

    # Characterise the start of the waveform using the l=m=2 psi4 multipole
    def characterize_start_end(this,turnon_width_in_cylcles=3):

        # Look for the l=m=2 psi4 multipole
        if len( this.ylm ):
            y22 = this.lm[2,2]['psi4']
        elif len( this.hlm ):
            y22 = this.lm[2,2]['strain']
        else:
            # If it doesnt exist in this.ylm, then load it
            y22 = this.load(lm=[2,2],output=True,dt=this.dt)

        #%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&#
        # Characterize the START of the waveform (pre-inspiral)      #
        #%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&#
        # Use the l=m=2 psi4 multipole to determine the waveform start
        # store information about the start of the waveform to the current object
        this.preinspiral = gwfcharstart( y22, shift=turnon_width_in_cylcles )
        # store the expected min frequency in the waveform to this object as:
        this.wstart = this.preinspiral.left_dphi
        this.startindex = this.preinspiral.left_index
        # Estimate the smallest orbital frequency relevant for this waveform using a PN formula.
        safety_factor = 0.90
        this.wstart_pn = safety_factor*2.0*pnw0(this.m1,this.m2,this.b)

        #%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&#
        # Characterize the END of the waveform (post-ringdown)       #
        #%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&%%&#
        this.postringdown = gwfcharend( y22 )
        # After endindex, the data is dominated by noise
        this.endindex = this.postringdown.right_index
        this.endindex_by_amplitude = this.postringdown.left_index
        this.endindex_by_frequency = this.postringdown.frequency_end_index

    # Clean the time domain waveform by removing junk radiation.
    def clean( this, method=None, crop_time=None ):

        # Default cleaning method will be smooth windowing
        if method is None:
            method = 'window'

        # ---------------------------------------------------------------------- #
        # A. Clean the start and end of the waveform using information from the
        #    characterize_start_end method
        # ---------------------------------------------------------------------- #

        if not this.__isclean__ :

            if method.lower() == 'window':

                # # Look for the l=m=2 psi4 multipole
                # y22_list = filter( lambda y: y.l==y.m==2, this.ylm )
                # # If it doesnt exist in this.ylm, then load it
                # if 0==len(y22_list):
                #     y22 = this.load(lm=[2,2],output=True,dt=this.dt)
                # else:
                #     y22 = y22_list[0]

                # NOTE that the l=m=2 multipole is used to define the windows below. See gwylm.characterize_start_end for more info.

                # Calculate the window to be applied using the starting information. The window nwill be aplied equally to all multipole moments. NOTE: language disambiguation -- a taper is the part of a window that varies from zero to 1 (or 1 to zero); a window may contain many tapers. Also NOTE that the usage of this4[0].ylm[0].t below is an arbitration -- any array of the dame dimentions could be used.
                # -- The above is calculated in the gwfcharstart class -- #
                # Extract the post-ringdown window
                preinspiral_window = this.preinspiral.window

                # Extract the post-ringdown window (calculated in the gwfcharend class)
                postringdown_window = this.postringdown.window

                # Construct the combined window
                window = preinspiral_window * postringdown_window

                # Apply WINDOW to children
                for y in this.ylm:
                    y.apply_window( window=window )
                for h in this.hlm:
                    h.apply_window( window=window )
                for f in this.flm:
                    f.apply_window( window=window )

            elif method.lower() == 'crop':

                # Crop such that the waveform daya starts abruptly
                from numpy import arange,double

                if (crop_time is None):
                    # If there is no crop time given, then use the low frequency value given by the nrutils start characterization time
                    mask = arange( this.startindex, this.ylm[0].n )
                elif isinstance(crop_time,(double,int,float)):
                    # Otherwise, use an input starting time
                    mask = this.ylm[0].t > crop_time

                # Apply MASK to children
                for y in this.ylm:
                    y.apply_mask( mask )
                for h in this.hlm:
                    h.apply_mask( mask )
                for f in this.flm:
                    f.apply_mask( mask )

            # Tag this object as clean
            this.__isclean__ = True

    # Reset each multipole object to its original state
    def reset(this):
        #
        for y in this.ylm:
            y.reset()
        for h in this.hlm:
            h.reset()

    # return a copy of the current object
    def copy(this):

        #
        from copy import deepcopy as copy
        return copy(this)


    #--------------------------------------------------------------------------------#
    # Calculate the luminosity if needed (NOTE that this could be calculated by default during calcstrain but isnt)
    #--------------------------------------------------------------------------------#
    def calcflm(this,           # The current object
                w22=None,       # Frequency used for FFI integration
                force=False,    # Force the calculation if it has already been performed
                verbose=False): # Let the people know

        # Make sure that the l=m=2 multipole exists
        if not ( (2,2) in this.lm.keys() ):
            msg = 'There must be a l=m=2 multipole prewsent to estimate the waveform\'s ringdown part.'
            error(msg,'gwylm.ringdown')

        # Determine whether or not to proceed with the calculation
        # Only proceed if the calculation has not been performed before and if the force option is False
        proceed = (not this.flm) or force
        if proceed:

            # Import useful things
            from numpy import array,double

            # If there is no w22 given, then use the internally defined value of wstart
            if w22 is None:
                # w22 = this.wstart
                # NOTE: here we choose to use the ORBITAL FREQUENCY as a lower bound for the l=m=2 mode.
                w22 = this.wstart_pn

            # Calculate the luminosity for all multipoles
            flm = []
            proceed = True
            for y in this.ylm:

                # Calculate the strain for each part of psi4. NOTE that there is currently NO special sign convention imposed beyond that used for psi4.
                w0 = w22 * double(y.m)/2.0 # NOTE that wstart is defined in characterize_start_end() using the l=m=2 Psi4 multipole.
                # Here, m=0 is a special case
                if 0==y.m: w0 = w22
                # Let the people know
                if this.verbose:
                    alert( magenta('w0(w22) = %f' % w0)+yellow(' (this is the lower frequency used for FFI method [arxiv:1006.1632v3])') )

                # Create the core waveform information
                t       =  y.t
                l_plus  =  ffintegrate( y.t, y.plus,  w0, 1 )
                l_cross =  ffintegrate( y.t, y.cross, w0, 1 )

                # Constrcut the waveform array for the news object
                wfarr = array( [ t, l_plus, l_cross ] ).T

                # Add the news multipole to this object's list of multipoles
                flm.append( gwf( wfarr, l=y.l, m=y.m, kind='$r\dot{h}_{%i%i}$'%(y.l,y.m) ) )

            else:

                msg = 'flm, the first integral of Psi4, will not be calculated because it has already been calculated for the current object'
                if verbose: warning(msg,'gwylm.calcflm')

            # Store the flm list to the current object
            this.flm = flm

        # Create a dictionary representation of the mutlipoles
        this.__curate__()

        # NOTE that this is the end of the calcflm method


    #--------------------------------------------------------------------------------#
    # Get a gwylm object that only contains ringdown
    #--------------------------------------------------------------------------------#
    def ringdown(this,              # The current object
                 T0 = None,         # Starting time relative to peak luminosity of the l=m=2 multipole
                 T1 = None,         # Maximum time
                 df = None,         # Optional df in frequency domain (determines time domain padding)
                 use_peak_strain = True,   # Toggle to use peak of strain rather than the peak of the luminosity
                 verbose = None):

        #
        from numpy import linspace,array,where
        from scipy.interpolate import InterpolatedUnivariateSpline as spline

        # Make sure that the l=m=2 multipole exists
        if not ( (2,2) in this.lm.keys() ):
            msg = 'There must be a l=m=2 multipole prewsent to estimate the waveform\'s ringdown part.'
            error(msg,'gwylm.ringdown')

        # Let the people know (about which peak will be used)
        if this.verbose or verbose:
            alert('Time will be listed relative to the peak of %s.'%cyan('strain' if use_peak_strain else 'luminosity'))

        # Use the l=m=2 multipole to estimate the peak location
        if use_peak_strain:
            # Only calculate strain if its not there already
            if (not this.hlm) : this.calchlm()
        else:
            # Redundancy checking (see above for strain) is handled within calcflm
            this.calcflm()

        # Retrieve the l=m=2 component
        ref_gwf = this.lm[2,2][  'strain' if use_peak_strain else 'news'  ]
        # ref_gwf = [ a for a in (this.hlm if use_peak_strain else this.flm) if a.l==a.m==2 ][0]

        #
        # peak_time = ref_gwf.t[ ref_gwf.k_amp_max ]
        peak_time = ref_gwf.intrp_t_amp_max

        # Handle T0 input
        if T0 is None:
            #
            this.calcflm()
            mu = 0.25
            dE = sum( [ f.amp**2 for f in this.flm ] )
            T0 = ref_gwf.t[ (dE<(mu*dE.max())) & (ref_gwf.t>peak_time) ][0] - peak_time
            #
            if T0 < 0: T0 = 20

        # Handle T1 Input
        if T1 is None:
            # NOTE that we will set T1 to be *just before* the noise floor estimate
            T_noise_floor = ref_gwf.t[this.postringdown.left_index] - peak_time
            # "Just before" means 95% of the way between T0 and T_noise_floor
            safety_factor = 0.45 # NOTE that this is quite a low safetey factor -- we wish to definitely avoid noise if possible. T1_min is implemented below just in case this is too strong of a safetey factor.
            T1 = T0 + safety_factor * ( T_noise_floor - T0 )
            # Make sure that T1 is at least T1_min
            T1_min = T0+60
            T1 = max(T1,T1_min)
            # NOTE that there is a chance that T1 chould be "too close" to T0
        elif T1 == 'end':
            # Deliberately use the end of the waveform
            T1 = ref_gwf.t[-1] - peak_time

        # Validate T1 Value
        if T1<T0:
            msg = 'T1=%f which is less than T0=%f. This doesnt make sense: the fitting region cannot end before it begins under the working perspective.'%(T1,T0)
            error(msg,'gwylm.ringdown')
        if T1 > (ref_gwf.t[-1] - peak_time) :
            msg = 'Input value of T1=%i extends beyond the end of the waveform. We will stop at the last value of the waveform, not at the requested T1.'%T1
            warning(msg,'gwylm.ringdown')
            T1 = ref_gwf.t[-1] - peak_time

        # Use its time series to define a mask
        a = peak_time + T0
        b = peak_time + T1
        n = 1+abs(float(b-a))/ref_gwf.dt
        t = linspace(a,b,n)

        #
        that = this.copy()
        that.__isringdownonly__ = True
        that.T0 = T0
        that.T1 = T1

        #
        def __ringdown__(wlm):
            #
            xlm = []
            for k,y in enumerate(wlm):
                # Create interpolated plus and cross parts
                plus  = spline(y.t,y.plus)(t)
                cross = spline(y.t,y.cross)(t)
                # Create waveform array
                wfarr = array( [t-peak_time,plus,cross] ).T
                # Create gwf object
                xlm.append(  gwf(wfarr,l=y.l,m=y.m,mf=this.mf,xf=this.xf,kind=y.kind,label=this.label,m1=this.m1,m2=this.m2,ref_scentry = this.__scentry__)  )
            #
            return xlm
        #
        that.ylm = __ringdown__( this.ylm )
        that.flm = __ringdown__( this.flm )
        that.hlm = __ringdown__( this.hlm )

        # Create a dictionary representation of the mutlipoles
        that.__curate__()

        #
        return that


    # pad each mode to a new_length
    def pad(this,new_length=None, apply=True, extend=True ):
        # Pad each mode
        ans = this if apply else this.copy()
        treset=True
        for z in this.lm:
            for k in this.lm[z]:
                ans.lm[z][k].pad( new_length=new_length, apply=apply, extend=extend )
                if treset:
                    ans.t = ans.lm[z][k].t
                    treset = False
        #
        if not apply:
            if len(ans.t)!=new_length:
                error('!!!')
            return ans

    # shift the time series
    def tshift( this, shift=0, apply=True ):
        # shift each mode
        ans = this if apply else this.copy()
        for z in ans.lm:
            for k in ans.lm[z]:
                ans.lm[z][k].tshift( shift=shift, apply=apply )
        #
        if not apply: return ans


    # Recompose the waveforms at a sky position about the source
    def recompose( this,
                   theta,
                   phi,
                   kind = None,
                   domain = None,       # only useful if output_array = True
                   select_lm = None,
                   output_array = None, # Faster as the gwf constructor is not called (e.g. related ffts are not taken)
                   verbose = None):

        # Validate the inputs
        if kind is None:
            msg = 'no kind specified for recompose calculation. We will proceed assuming that you desire recomposed strain. Please specify the desired kind (e.g. strain, psi4 or news) you wish to be output as a keyword (e.g. kind="news")'
            # warning( msg, 'gwylm.recompose' )
            kind = 'strain'
        if domain is None:
            msg = 'no domain specified for recompose calculation. We will proceed assuming that you desire recomposed time domain data. Please specify the desired domain (e.g. time or freq) you wish to be output as a keyword (e.g. domain="freq")'
            # warning( msg, 'gwylm.recompose' )
            domain = 'time'

        # if it is desired to work with arrays
        if output_array:
            #
            if (kind is None) or (domain is None):
                error('When recomposing arrays, BOTH domain and kind keyword inputs must be given.')
            ans = this.__recompose_array__( theta, phi, kind, domain, select_lm=select_lm, verbose=verbose )
        else: # if it desired to work with gwf objects (this is time domain recomposition followed by gwf construction)
            #
            ans = this.__recompose_gwf__( theta, phi, kind=kind, select_lm=select_lm, verbose=verbose )

        # Return the answer
        return ans

    # Enforce M_RELATIVE_SIGN_CONVENTION
    def __enforce_m_relative_phase_orientation__(this):

        # Import usefuls
        from numpy import arange,sign,diff,unwrap,angle,amax,isnan,amin,log,exp,std,median,mod,mean
        from scipy.stats.mstats import mode
        from scipy.version import version as scipy_version
        thisfun=inspect.stack()[0][3]

        # Use the 2,2, multipole just after wstart to determine initial phase direction
        mask = arange(this.startindex,this.startindex+50)
        dphi = this[2,2]['psi4'].dphi[mask]
        m=2

        if int(scipy_version.split('.')[1])<16:
            # Account for old scipy functionality
            external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( dphi ) )[0][0]
            initially_msign_matches_wsign = sign(m) == mode( sign( dphi ) )[0][0]
        else:
            # Account for modern scipy functionality
            external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( dphi ) ).mode[0]
            initially_msign_matches_wsign = sign(m) == mode( sign( dphi ) ).mode[0]
        # if initially_msign_matches_wsign: alert('## initall, m and td freq have same sign.')
        this.external_sign_convention = external_sign_convention

        if this.M_RELATIVE_SIGN_CONVENTION != this.external_sign_convention:
            # Let the people know what is happening.
            msg = yellow('[Verify stage] Re-orienting waveform phase')+' to be consistent with internal sign convention for Psi4, where sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]).' % this.M_RELATIVE_SIGN_CONVENTION + ' Note that the internal sign convention is defined in ... nrutils/core/__init__.py as "M_RELATIVE_SIGN_CONVENTION". This message has appeared becuase the waveform is determined to obey and sign convention: sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]). Note the appearance of the initial z angular momentum, this.L[-1].'%(this.external_sign_convention)
            thisfun=inspect.stack()[0][3]
            warning( msg, verbose=this.verbose )
            #
            for l,m in this.lm:
                for kind in this[l,m]:
                    y = this[l,m][kind]
                    wfarr = y.wfarr
                    wfarr[:,2] *= -1
                    y.setfields( wfarr )
                    this[l,m][kind] = y


        # for l,m in this.lm:
        #     for kind in this[l,m]:
        #         y = this[l,m][kind]
        #         wfarr = y.wfarr
        #         wfarr[:,2] *= -this.M_RELATIVE_SIGN_CONVENTION
        #         y.setfields( wfarr )
        #         this[l,m][kind] = y

        # # Try to determine the sign convention used to define phase. Note that this will be determined only once for the current object based on the l=m=2 multipole.
        # if this.external_sign_convention is None:
        #     msk_ = y_.amp > 0.0001*amax(y_.amp)
        #     # msk_ = y_.amp > 0.01*amax(y_.amp)
        #     if int(scipy_version.split('.')[1])<16:
        #         # Account for old scipy functionality
        #         external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( y_.dphi[msk_] ) )[0][0]
        #         initially_msign_matches_wsign = sign(m) == mode( sign( y_.dphi[msk_] ) )[0][0]
        #     else:
        #         # Account for modern scipy functionality
        #         external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( y_.dphi[msk_] ) ).mode[0]
        #         initially_msign_matches_wsign = sign(m) == mode( sign( y_.dphi[msk_] ) ).mode[0]
        #     if initially_msign_matches_wsign: alert('## initall, m and td freq have same sign.')
        #     this.external_sign_convention = external_sign_convention
        #
        # if this.M_RELATIVE_SIGN_CONVENTION != this.external_sign_convention:
        #     wfarr[:,2] = -wfarr[:,2]
        #     y_ = mkgwf(wfarr)
        #     # Let the people know what is happening.
        #     msg = yellow('Re-orienting waveform phase')+' to be consistent with internal sign convention for Psi4, where sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]).' % this.M_RELATIVE_SIGN_CONVENTION + ' Note that the internal sign convention is defined in ... nrutils/core/__init__.py as "M_RELATIVE_SIGN_CONVENTION". This message has appeared becuase the waveform is determioned to obey and sign convention: sign(dPhi/dt)=%i*sign(m)*sign(this.L[-1]). Note the appearance of the initial z angular momentum, this.L[-1].'%(this.external_sign_convention)
        #     thisfun=inspect.stack()[0][3]
        #     warning( msg, verbose=this.verbose )

        #

        #@
        return None

    # recompose individual arrays for a select data type (psi4, strain or news)
    def __recompose_array__( this,theta,phi,kind,domain,select_lm=None,verbose=False ):
        '''
        Recompose individual arrays for a select data type (psi4, strain or news)
        '''

        # Set default for select_lm
        select_lm = this.__input_lmlist__ if select_lm is None else select_lm

        # Construct functions which handle options
        fd_wfarr_dict_fun = lambda k: { lm:this.lm[lm][k].fd_wfarr for lm in select_lm }
        td_wfarr_dict_fun = lambda k: { lm:this.lm[lm][k].wfarr for lm in select_lm }
        wfarr_dict_fun    = lambda d,k: fd_wfarr_dict_fun(k) if d in ('fd','freq','fequency','f') else td_wfarr_dict_fun(k)

        #  Get desired waveform array
        wfarr_dict = wfarr_dict_fun(domain,kind)

        #
        error('There\'s a bug in this workflow that cases the spectra of h+/x to not obey conjugate symmetry!!')
        # Recompose using low level function in basics.py
        recomposed_wfarr = recompose_wfarrs( wfarr_dict, theta, phi )

        # Return answer
        ans = recomposed_wfarr
        return ans

    #
    def getframe(this,frame_type,verbose=None,domain=None):

        #
        if domain is None: domain = 'time'

        #
        if not (domain in ('t','time')):
            error('this method only handles time domain frames for the time being')

        #
        valid_frames = ('coprecessing','cp','jt','jinit')

        #
        if not (frame_type in valid_frames):
            error( 'invalid/unknown frame type given' )


    #
    def __calc_initial_j_frame__(this):
        '''
        Rotate multipoles such that initial J is parallel to z-hat
        '''

        # Import usefuls
        from numpy import arccos,arctan2,array,linalg,cos,sin,dot,zeros,ones
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS

        J_norm = linalg.norm(this.J)
        thetaJ = arccos(this.J[2]/J_norm)
        phiJ   = arctan2(this.J[1],this.J[0])

        # Define gamma and beta accordingly
        beta  = -thetaJ
        gamma = -phiJ

        # Define zeta0 (i.e. -alpha) such that L is along the y-z plane at the initial time step
        L_new = rotate3 ( this.L1 + this.L2, 0, beta , gamma )
        zeta0 = arctan2( L_new.T[1], L_new.T[0] )
        alpha = -zeta0

        # Bundle rotation angles
        angles = [ alpha, beta, gamma ]

        # perform rotation
        that = this.__rotate_frame_at_all_times__(angles)

        #
        that.frame = 'J-initial'

        #
        return that

    #
    def __calc_initial_l_frame__(this):
        '''
        Rotate multipoles such that initial L is parallel to z-hat
        '''

        # Import usefuls
        from numpy import arccos,arctan2,array,linalg,cos,sin,dot,zeros,ones
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS

        L_norm = linalg.norm(this.L)
        thetaL = arccos(this.L[2]/L_norm)
        phiL   = arctan2(this.L[1],this.L[0])

        # Define gamma and beta accordingly
        beta  = -thetaL
        gamma = -phiL

        # Define zeta0 (i.e. -alpha) such that J is along the y-z plane at the initial time step
        J_new = rotate3 ( this.J, 0, beta , gamma )
        zeta0 = arctan2( J_new.T[1], J_new.T[0] )
        alpha = -zeta0

        # Bundle rotation angles
        angles = [ alpha, beta, gamma ]

        # perform rotation
        that = this.__rotate_frame_at_all_times__(angles)

        #
        that.frame = 'L-initial'

        #
        return that


    #
    def __calc_j_of_t_frame__(this,verbose=None):

        #
        from numpy.linalg import norm
        from numpy import arccos,arctan2,array,cos,sin,dot,zeros,ones
        from scipy.interpolate import InterpolatedUnivariateSpline as IUS

        #
        this.__calc_radiated_quantities__(use_mask=False)

        # get time series for radiated quantities
        t = this.remnant['time_used']

        #
        J = this.remnant['J']
        J_norm = norm(J, axis=1)
        thetaJ = zeros( len(J_norm) )
        phiJ   = zeros( len(J_norm) )
        for k in range ( len(J_norm) ):
            thetaJ[k] = arccos(J[k,2]/J_norm[k])
            phiJ[k]   = arctan2(J[k,1],J[k,0])
        #
        phiJ_spl  = IUS(t, phiJ, k=5)
        dp_dt_spl = phiJ_spl.derivative()
        dp_dt     = dp_dt_spl(t)
        # calculate zeta according to the minimal rotation condition: Eq. 18 of arxiv::1110.2965
        dz_dt = -dp_dt*cos(thetaJ)
        # calculate zeta
        dz_dt_spl = IUS(t, dz_dt, k=5)
        zeta_spl  = dz_dt_spl.antiderivative()
        zeta      = zeta_spl(t)

        # Define gamma and beta accordingly
        beta  = -thetaJ
        gamma = -phiJ
        # Define alpha such that initial L is aling x
        L_new = rotate3 ( this.L1 + this.L2, 0, beta[0], gamma[0] )
        zeta0 = arctan2( L_new.T[1], L_new.T[0] )
        alpha = -( zeta - zeta[0] + zeta0 )

        # Bundle rotation angles
        angles = [ alpha, beta, gamma ]

        #
        that = this.__rotate_frame_at_all_times__(angles)

        #
        that.frame = 'J(t)'

        #
        that.__calc_radiated_quantities__(use_mask=False)


        # Perform rotation
        return that


    # output corotating waveform
    def __calc_coprecessing_frame__(this,verbose=None):

        '''
        Output gwylm object in coprecessing frame, where the optimal emission axis is always along z
        '''

        #
        from nrutils.manipulate.rotate import gwylm_radiation_axis_workflow

        #
        if verbose is None: verbose = this.verbose

        #
        foo = gwylm_radiation_axis_workflow(this,plot=False,save=False,verbose=verbose)

        #
        if verbose: alert('Storing radiation axis information to this.radiation_axis_info')
        this.radiation_axis_info = foo

        #
        alpha = foo.radiation_axis['td_alpha']
        beta = foo.radiation_axis['td_beta']
        gamma = foo.radiation_axis['td_gamma']

        #
        that = this.__rotate_frame_at_all_times__( [gamma,-beta,alpha] )
        that.previous_radiation_axis_info = foo


        #
        return that

    # Recompose the waveforms at a sky position about the source
    # NOTE that this function returns a gwf object
    def __recompose_gwf__( this,         # The current object
                           theta,        # The polar angle
                           phi,          # The anzimuthal angle
                           kind=None,
                           select_lm=None,
                           verbose=False ):

        #
        from numpy import dot,array,zeros

        #
        select_lm = this.__input_lmlist__ if select_lm is None else select_lm

        # Create Matrix of Multipole time series
        def __recomp__(alm,kind=None):

            M = zeros( [ alm[0].n, len(this.__input_lmlist__) ], dtype=complex )
            Y = zeros( [ len(this.__input_lmlist__), 1 ], dtype=complex )
            # Seed the matrix as well as the vector of spheroical harmonic values
            for k,a in enumerate(alm):
                lm = (a.l,a.m)
                if (lm in this.__input_lmlist__) and (lm in select_lm):
                    M[:,k] = a.y
                    Y[k] = sYlm(-2,a.l,a.m,theta,phi)
            # Perform the matrix multiplication and create the output gwf object
            Z = dot( M,Y )[:,0]
            wfarr = array( [ alm[0].t, Z.real, Z.imag ] ).T
            # return the ouput
            return gwf( wfarr, kind=kind, ref_scentry = this.__scentry__ )

        #
        if kind=='psi4':
            y = __recomp__( this.ylm, kind=r'$rM\,\psi_4(t,\theta,\phi)$' )
        elif kind=='strain':
            y = __recomp__( this.hlm, kind=r'$r\,h(t,\theta,\phi)/M$' )
        elif kind=='news':
            y = __recomp__( this.flm, kind=r'$r\,\dot{h}(t,\theta,\phi)/M$' )

        #
        return y

    # Phase shift each mode according to a rotation of the orbital plane
    def rotate( this, dphi=0, apply=True, dpsi=0, verbose=True, fast=False ):
        '''Phase shift each mode according to a rotation of the orbital plane'''
        # Import useful things
        from numpy import array,sin,cos,arctan2,dot,sign
        # For all multipole sets
        ans = this if apply else this.copy()
        for j in ans.lm:
            for k in ans.lm[j]:
                m = ans.lm[j][k].m
                ans.lm[j][k].shift_phase( m * dphi + dpsi*sign(m), apply=True, fast=fast )
        # Apply rotation to position metadata
        if 'R1' in ans.__dict__:
            R1_ = array(ans.R1)
            R2_ = array(ans.R2)
            M = array([[ cos(dpsi), -sin(dpsi), 0 ],
                       [ sin(dpsi),  cos(dpsi), 0 ],
                       [         0,          0, 1 ]])
            R1_ = dot( M, R1_ )
            R2_ = dot( M, R2_ )
            #
            for k in range(len(R1_)):
                ans.R1[k] = R1_[k]
                ans.R2[k] = R2_[k]
        #
        if verbose: warning('Note that this method only affects waveforms, meaning that rotations are not back propagated to metadata: spins, component positions etc. This is for future work. Call rotate with verbose=False to disable this message.')
        if not apply:
            return ans
        else:
            return None

    #
    def deletelm( this, lm ):
        # remove an lm node from this object
        error('method not implemented')
        return None

    # Find the polarization and orbital phase shifts that maximize the real part
    # of  gwylm object's (2,2) and (2,1) multipoles at merger (i.e. the sum)
    def selfalign( this, ref_gwylmo=None, plot=False, apply=True, n=13, verbose=False, v=1 ):
        '''
        Find the polarization and orbital phase shifts that maximize the real part
        of  gwylm object's (2,2) and (2,1) multipoles at merger (i.e. the sum)
        '''

        # Let the people know
        if verbose: alert('Appling the polarization and orbital phase shifts that maximize the real part of the gwylm objects (2,2) and (2,1) multipoles at merger (i.e. the sum)')
        # Choose a region around ringdown to use for teh opimization uness give a reference gwylmo to use instead
        u = this.ringdown(T0=0,T1=20) if ref_gwylmo is None else ref_gwylmo
        # Maximize
        if v == 1:
            dphi,dpsi = betamax(u,plt=plot,opt=True,n=n, verbose=verbose)
            print dphi,dpsi
        else:
            print '** Using test version betamax2'
            dphi,dpsi = betamax2(u,plt=plot,opt=True,n=n, verbose=verbose)
            print dphi,dpsi
        # Rotate self
        ans = this.rotate( dphi=dphi, dpsi=dpsi, verbose=verbose, apply=apply )

        # #
        # from numpy import pi,angle
        # #
        # u = this.ringdown(T0=0,T1=20) if ref_gwylmo is None else ref_gwylmo
        # dphi = -angle(u.lm[2,1]['psi4'].phi[0])
        # ans = this.rotate( dphi=dphi, dpsi=0, verbose=verbose, apply=apply )
        # dpsi = -angle(u.lm[2,2]['psi4'].phi[0])
        # ans = this.rotate( dphi=0, dpsi=dpsi, verbose=verbose, apply=apply )

        #
        return ans

    # Rotate the orbital phase of the current set to align with a reference gwylm object
    def align( this, that,
               plot=False,
               apply=True,
               verbose=True,
               lm=None,
               return_match=False,
               kind=None,
               plot_labels=None ):

        '''Rotate the orbital phase and polarization of the current set to align with a reference gwylm object'''

        # Import useful things
        from numpy import array,argmax,pi,linspace,angle
        from scipy.optimize import minimize

        #
        kind = 'psi4' if kind is None else kind
        if not kind in ('strain','psi4'): error('unknown kind given')

        # Define a shortahnd
        (a,b) = (this,that) if apply else (this.copy,that.copy)

        # Define data holders and the range of orbital phase to brture force
        ab_list = []
        dphi_range = pi*linspace(-1,1,19)
        # Evaluate the nrxcorr over the orbital phase brute force points
        # and convert result into numpy array
        if verbose: alert('Performing sky-averaged match (no noise curve) to estimate optimal shift in orbital phase.')
        ab_list = [ a.nrxcorr(b,dphi,lm=lm,kind=kind) for dphi in dphi_range ]
        ab = array(ab_list)
        # Perform numerical optimization
        if verbose: alert('Obtaining numerical estimate of optimal shift in orbital phase.')
        action = lambda du: -abs( a.nrxcorr(b,du,lm=lm,kind=kind) )
        guess_dphi = dphi_range[ argmax( abs(ab) ) ]
        Q = minimize( action, guess_dphi, bounds=[(-pi,pi)] )
        # Extract optimal orbital phase shift & calculate optimal polzarization shift
        dphi_opt = Q.x[0]
        dpsi_opt = -angle( a.nrxcorr(b,dphi_opt,lm=lm,kind=kind) )
        #
        a.rotate( dphi_opt, dpsi=dpsi_opt, verbose=verbose, apply=True )

        #
        if verbose: alert('(dphi_opt,dpsi_opt) = (%1.4f,%1.4f)'%(dphi_opt,dpsi_opt))

        #
        if plot:

            from scipy.interpolate import interp1d as spline
            # Setup plotting backend
            import matplotlib as mpl
            from mpl_toolkits.mplot3d import axes3d
            mpl.rcParams['lines.linewidth'] = 0.8
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.size'] = 12
            mpl.rcParams['axes.labelsize'] = 20
            mpl.rcParams['axes.titlesize'] = 20
            from matplotlib import rc
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            from matplotlib.pyplot import plot,axes,xlabel,ylabel,xlim,ylim,figure,title

            dphis = pi*linspace(-1,1,2e2).T
            M = lambda dphi_: spline(dphi_range,ab.real, kind='cubic')(dphi_) + 1j*spline(dphi_range,ab.imag, kind='cubic')(dphi_)
            ms = M(dphis)

            fig = figure( figsize=2*array([5,3]) )

            plot( dphi_range, abs(ab), linewidth=4, color='k',alpha=0.1 )
            plot( dphi_range, ab.real, linewidth=4, color='k',alpha=0.1 )
            plot( dphi_range, ab.imag, linewidth=4, color='k',alpha=0.1 )

            plot( dphis, abs(ms) )
            plot( dphis, ms.real, alpha=0.5 )
            plot( dphis, ms.imag, alpha=0.5 )
            plot( dphi_opt, abs(M(dphi_opt)), 'or' )

            title(r'$(d\phi,d\psi,max) = (%1.4f,%1.4f,%1.4f)$'%(  dphi_opt,dpsi_opt, abs(M(dphi_opt))  ) )
            xlabel(r'$d\phi$')
            ylabel(r'$\langle u,v \rangle$')

            xlim(lim(dphis))

            #
            for j in a.lm:
                if j in b.lm:
                    b.lm[j][kind].plot( ref_gwf = a.lm[j][kind],labels= ('u','v') if plot_labels is None else plot_labels )

        #
        if return_match:
            return -Q.fun
        else:
            if not apply: return a,b

    # Given a reference gwylm, ensure that there is a common dt
    def dtalign(this,yref,apply=True,kind=None):
        '''Given a reference gwylm, ensure that there is a common dt'''
        kind = 'strain' if kind is None else kind
        if not kind in ('strain','psi4'): error('unknown kind given')
        (that,zref) = (this,yref) if apply else (this.copy(),yref.copy())
        if kind in ('psi4'):
            dt1 = that.ylm[0].dt
            dt2 = yref.ylm[0].dt
        elif kind in ('strain'):
            dt1 = that.hlm[0].dt
            dt2 = yref.hlm[0].dt
        if dt1!=dt2: ( that if dt2<dt1 else zref ).setdt( min([dt1,dt2]) )
        return that,zref

    # Change the dt of the current gwylm via interpolation
    def setdt(this,dt,apply=True):
        '''Change the dt of the current gwylm via interpolation'''
        ans = this if apply else this.copy()
        for j in ans.lm:
            for k in ans.lm[j]:
                ans.lm[j][k].interpolate( dt=dt )
        if not apply: return ans

    # Given a reference gwylm, pad the current object and perhaps the reference to the same length
    def lengthalign(this,yref,apply=True,kind=None):
        '''Given a reference gwylm, pad the current object and perhaps the reference to the same length'''
        #
        kind = 'strain' if kind is None else kind
        if not kind in ('strain','psi4'): error('unknown kind given')
        that,zref = this.dtalign(yref,apply=apply)
        if kind in ('psi4'):
            l1 = that.ylm[0].n
            l2 = zref.ylm[0].n
        elif kind in ('strain','h'):
            l1 = that.hlm[0].n
            l2 = zref.hlm[0].n

        if l1!=l2: ( that if l2>l1 else zref ).pad( max([l1,l2]) )
        return that,zref

    # Given a reference gwylm, align the peak to that of a reference waveform
    def tpeakalign(this,yref,apply=True,kind=None):
        '''Given a reference gwylm, align the peak to that of a reference waveform'''
        kind = 'strain' if kind is None else kind
        if not kind in ('strain','psi4'): error('unknown kind given')
        that,zref = this.lengthalign( yref, apply=apply, kind=kind )
        shift = -that.lm[2,2][kind].intrp_t_amp_max + zref.lm[2,2][kind].intrp_t_amp_max
        that.tshift( shift=shift )
        if not apply: return that,zref

    # Compute the simple vector inner-product (sky averaged overlap with no noise curve) between one gwylm  and another
    def nrxcorr(this,     # The current object
                yref,     # reference gwylm
                dphi=0,   # rotation of orbital phase to apply to current object
                dpsi=0,   # rotation of polarization to apply to current object
                verbose=True,
                kind=None,
                lm = None): # which modes to use
        '''
        # Compute the simple vector inner-product (sky averaged overlap with no noise curve) between one gwylm  and another.

        Inputs:
        ---
        yref,   # reference gwylm
        dphi,   # rotation of orbital phase to apply to current object
        dpsi,   # rotation of polarization to apply to current object
        lm,     # which multipoles to use for match - useful for investigating degeneracies in dpsi & dphi

        Output:
        ---
        x,      # sky avergaed and normalized inner-product
        '''
        # if verbose: warning('this is not a proper match function, but an add-hock flat psd sky averaged match; for proper matches see nrutils.analysis.match')
        # Import useful things
        from numpy import sqrt
        #
        kind = 'psi4' if kind is None else kind
        #
        that,zref = this.lengthalign(yref,apply=False,kind=kind)
        lm  = zref.__lmlist__ if lm is None else lm
        # Define a simple inner product
        def prod(a,b): return sum(a.conj()*b)
        # Calculate a sky averaged and normalized inner product
        x,U,V,N = 0,0,0,len(lm)
        that = that.rotate(dphi,apply=False,dpsi=dpsi,verbose=False)

        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#
        # The code below normalizes the match for the entire sky average
        proceedfun = lambda ll,mm: True if lm is None else ((ll,mm) in lm)
        for k in range(N):
            if kind=='psi4':
                l,m = zref.ylm[k].l,zref.ylm[k].m
            else:
                l,m = zref.hlm[k].l,zref.hlm[k].m
            proceed = proceedfun(l,m) # True if lm is None else (l,m) in lm
            if proceed:
                u,v = zref.lm[l,m][kind].y, that.lm[l,m][kind].y
                U += prod(u,u); V += prod(v,v)
        for k in range(N):
            if kind=='psi4':
                l,m = zref.ylm[k].l,zref.ylm[k].m
            else:
                l,m = zref.hlm[k].l,zref.hlm[k].m
            proceed = proceedfun(l,m) # True if lm is None else (l,m) in lm
            if proceed:
                u,v = zref.lm[l,m][kind].y, that.lm[l,m][kind].y
                u_ = u/sqrt(U); v_ = v/sqrt(V)
                x += prod( u_,v_ ) if m>=0 else prod( u_,v_ ).conj()
        #--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--%%--#

        # # The code below normalizes the match for each mode
        # for k in range(N):
        #     l,m = zref.ylm[k].l,zref.ylm[k].m
        #     proceed = True if lm is None else (l,m) in lm
        #     if proceed:
        #         u,v = zref.lm[l,m]['psi4'].y, that.lm[l,m]['psi4'].y
        #         u_ = u/sqrt(prod(u,u)); v_ = v/sqrt(prod(v,v))
        #         x += ( prod( u_,v_ ) if m>=0 else prod( u_,v_ ).conj() )/N

        # Return the answer
        return x

    # Extrapolate to infinite radius: http://arxiv.org/pdf/1503.00718.pdf
    def extrapolate(this,method=None):

        msg = 'This method is under development and cannot currently be used.'
        error(msg)

        # If the simulation is already extrapolated, then do nothing
        if this.__isextrapolated__:
            # Do nothing
            print
        else: # Else, extrapolate
            # Use radius only scaling
            print

        return None

    # Given some time and set of euler angles, rotate all multipole data ... and possibly initial position, spin, and final spin.
    def __rotate_frame_at_all_times__( this,                        # The current object
                                       euler_alpha_beta_gamma,      # List of euler angles
                                       ref_orientation = None ):    # A reference orienation (useful for BAM)

        '''
        Given some time and set of euler angles, rotate all multipole data ... and possibly initial position, spin, and final spin.
        '''

        # Import usefuls
        from numpy import arccos,dot,ndarray,array

        # Perform roations for all kinds
        kinds = ['strain','psi4','news']

        # Create the output object -- its multipole data (and other?) will be replaced by rotated versions
        that = this.copy()

        #
        if not ( ref_orientation is None ) :
            error('The use of "ref_orientation" has been depreciated for this function.')

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # Rotate multipole data
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # For all kinds
        for kind in kinds:

            # For all multipoles
            for lm in this.lm:

                # Create list of this objects gwfs with the same kind and the same l
                like_l_multipoles = []
                for lp,mp in this.lm:
                    l,m = lm
                    if lp == l:
                        # print '>> (l,m)=(%i,%i)'%(l,m)
                        # print '>> (lp,mp)=(%i,%i)\n'%(lp,mp)
                        like_l_multipoles.append( this.lm[lp,mp][kind] )

                # Rotate the current multipole
                rotated_gwf = this.lm[lm][kind].__rotate_frame_at_all_times__( like_l_multipoles, euler_alpha_beta_gamma, ref_orientation )

                # Store it to the output gwylm object
                that.lm[lm][kind] = rotated_gwf

        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # Rotate related metadata??
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # * final spin vector
        alpha,beta,gamma = euler_alpha_beta_gamma

        # Test for arrays
        angles_are_arrays = isinstance( alpha, ndarray ) and isinstance( beta, ndarray ) and isinstance( gamma, ndarray )

        # Enforce arrays
        if not angles_are_arrays:
            alpha = array([alpha])
            beta = array([beta])
            gamma = array([gamma])


        R = lambda X,k: rotate3( X, alpha[k], beta[k], gamma[k] )
        that.Sf = R( this.Sf, -1 ); that.Xf = R( this.Xf, -1 )

        # * initial spins
        that.S1 = R( this.S1, 0 ); that.S2 = R( this.S2, 0 )
        that.X1 = R( this.X1, 0 ); that.X2 = R( this.X2, 0 )
        # * initial angular momenta
        that.L1 = R( this.L1, 0 ); that.L2 = R( this.L2, 0 )
        # * initial positions / position time series / maybe velocities
        that.R1 = R( this.R1, 0 ); that.R2 = R( this.R2, 0 )
        # * others
        that.S = R( this.S, 0 ); that.J = R( this.J, 0 );
        that.L = R( this.L, 0 )


        # Rotate system radiated and remnant quantities
        if not ( 'remnant' in this.__dict__ ) : this.__calc_radiated_quantities__(use_mask=False)
        that.old_remnant = copy.deepcopy(this.remnant)
        that.old_radiated = copy.deepcopy(this.radiated)

        # NOTE that the "old" quantities will be correct values for non-intertial (dynamical) angles

        for key in this.remnant:
            if isinstance(this.remnant[key],ndarray):
                # print key, len(this.remnant[key].shape)
                if this.remnant[key].shape[-1] == 3:
                    if len(alpha) == len(this.remnant['time_used']):
                        for k in range(len(alpha)):
                            that.old_remnant[key][k,:] = R( this.remnant[key][k,:], k )
                    elif len(alpha)==1:
                        for k in range( that.old_remnant[key].shape[0] ):
                            that.old_remnant[key][k,:] = R( this.remnant[key][k,:], 0 )
                    else:
                        warning('cannot rotate radiated quantities, length mismatch: len alpha is %i, but times are %i'%(len(alpha),len(this.remnant['time_used'])))
                        print alpha

        for key in this.radiated:
            if isinstance(this.radiated[key],ndarray):
                # print key, len(this.radiated[key].shape)
                if (this.radiated[key].shape[0] > 1) and (this.radiated[key].shape[-1] == 3):
                    if len(alpha) == len(this.radiated['time_used']):
                        if len(this.radiated[key].shape)>1:
                            for k in range(len(alpha)):
                                that.old_radiated[key][k,:] = R( this.radiated[key][k,:], k )
                        else:
                            if key=='J0':
                                that.old_radiated[key] = R( this.radiated[key], 0 )
                    elif len(alpha)==1:
                        if len(this.radiated[key].shape)>1:
                            for k in range( that.old_radiated[key].shape[0] ):
                                that.old_radiated[key][k,:] = R( this.radiated[key][k,:], 0 )
                        else:
                            that.old_radiated[key] = R( this.radiated[key], 0 )
                    else:
                        warning('cannot rotate radiated quantities, length mismatch: len alpha is %i, but times are %i'%(len(alpha),len(this.radiated['time_used'])))
                        print alpha

        #
        alert('Note that metadata at the scentry level (i.e. this.__scentry__) have not been rotated, but this.Sf, this.R1 and others have been rotated. This includes radiated and remnant quantities.')

        # Return answer
        return that

    # Estimate Remnant BH mass and spin from gwylm object. This is done by "brute" force here (i.e. an actual calculation), but NOTE that values for final mass and spin are Automatically loaded within each scentry; However!, some of these values may be incorrect -- especially for BAM sumulations. Here we make a rough estimate of the remnant mass and spin based on a ringdown fit.
    def brute_masspin( this,              # IMR gwylm object
                       T0 = 20,                 # Time relative to peak strain to start ringdown
                       T1 = None,               # Time relative to peak lum where ringdown ends (if None, gwylm.ringdown sets its value to the end of the waveform approx at noise floor)
                       apply_result = False,    # If true, apply result to input this object
                       verbose = False ):       # Let the people know
        '''Estimate Remnant BH mass and spin from gwylm object. This is done by "brute"
        force here (i.e. an actual calculation), but NOTE that values for final mass
        and spin are Automatically loaded within each scentry; However!, some of
        these values may be incorrect -- especially for BAM sumulations. Here we make
        a rough estimate of the remnant mass and spin based on a ringdown fit.'''

        # Import useful things
        thisfun='gwylm.brute_masspin'
        from scipy.optimize import minimize
        from nrutils import jf14067295,Mf14067295,remnant
        from kerr import qnmfit

        # Validate first input type
        is_number = isinstance(this,(float,int))
        is_gwylm = False if is_number else 'gwylm'==this.__class__.__name__
        if not is_gwylm:
            msg = 'First input  must be member of gwylm class from nrutils.'
            error(msg)

        # Get the ringdown part starting from 20M after the peak luminosity
        g = this.ringdown(T0=T0,T1=T1)
        # Define a work function
        def action( Mfxf ):
            # NOTE that the first psi4 multipole is referenced below.
            # There was only one loaded here, s it has to be for l=m=2
            f = qnmfit(g.lm[2,2]['psi4'],Mfxf=Mfxf)
            # f = qnmfit(g.ylm[0],Mfxf=Mfxf)
            return f.frmse

        # Use PhenomD fit for guess
        eta = this.m1*this.m2/((this.m1+this.m2)**2)
        chi1, chi2 = this.S1[-1]/(this.m1**2), this.S2[-1]/(this.m2**2)
        # guess_xf = jf14067295( this.m1,this.m2,chi1,chi2 )
        # guess_Mf = Mf14067295( this.m1,this.m2,chi1,chi2 )
        guess_Mf,guess_xf = remnant(this.m1,this.m2,this.chi1,this.chi2)
        guess = (guess_Mf,guess_xf)

        # perform the minization
        # NOTE that mass is bound on (0,1) and spin on (-1,1)
        Q = minimize( action,guess, bounds=[(1-0.999,1),(-0.999,0.999)] )

        # Extract the solution
        mf,xf = Q.x

        # Apply to the input gwylm object if requested
        if apply_result:
            this.mf = mf
            this.xf = xf
            this.Xf = this.Sf / (mf*mf)
            attr = [ 'ylm', 'hlm', 'flm' ]
            for atr in attr:
                for y in this.__dict__[atr]:
                    y.mf, y.xf = mf, xf
                    if ('Sf' in y.__dict__) and ('Xf' in y.__dict__):
                        y.Xf = y.Sf / (mf*mf)

        # Return stuff, including the fit object
        return mf,xf,Q


    # Estimate the energy radiated for the current collection of GW multipoles
    def __calc_radiated_quantities__(this,              # The current object
                                     use_mask = True,   # Toggle for chopping of noisey data. NOTE use_mask = False is useful if you need radiated quantities of the same length as the original waveforms
                                     ref_orientation = None,
                                     verbose=False):    # Toggle for letting the people know

        ''' Reference: https://arxiv.org/pdf/0707.4654.pdf '''

        # Import usefuls
        from numpy import trapz,pi,arange,isfinite,vstack,array,ones,sign

        # Construct a mask of useable data (OPTIONAL)
        if use_mask:
            mask = arange(this.startindex,this.endindex_by_frequency+1)
        else:
            mask = arange( len(this.t) )

        #
        if ref_orientation is None: ref_orientation = ones(3)

        # Since the mask will be optinal, let's use the hypothtical end value of the mask as a reference for the end of the waveform (i.e. before noise dominates)
        end_index = -1 if use_mask else this.endindex_by_frequency

        # Energy Raditated (Eq. 3.8)
        if verbose: alert('Calculating radiated energy, E.')
        if len(this.hlm)==0: this.calchlm()
        if len(this.flm)==0: this.calcflm()
        dE = (1.0/(16*pi)) * sum( [ f.amp**2 for f in this.flm ] )
        E0 = 1-this.madm # NOTE: this assumes unit norm for intial space-time energy
        if not isfinite(this.madm):
            E0 = 0
            warning('non-finite ADM mass given for this object; therefore, an initial radiated energy of 0 will be assumed to be valid for the start of the simulation.')
        E = E0 + spline_antidiff( this.t[mask],dE[mask],k=3 )
        E = E - E[end_index] + (1-this.mf) # Enforce consistency with final mass
        # Store radiated Quantities
        this.radiated = {}
        this.radiated['dE/dt'] = dE
        this.radiated['E'] = E
        this.radiated['time_used'] = this.t[mask]
        this.radiated['mask'] = mask
        if verbose: alert('Calculating radiated angular momentum, J.')
        this.radiated['J0'] = (this.S1 + this.S2) + (this.L1 + this.L2)
        this.radiated['J'] = this.__calc_radiated_angular_momentum__(mask)
        this.radiated['J_sim_start'] = this.Sf - this.radiated['J'][end_index,:]
        if verbose: alert('Calculating radiated linear momentum, P.')
        this.radiated['P'] = this.__calc_radiated_linear_momentum__(mask)

        #
        if not ( ref_orientation is None ):
            #
            this.radiated['J'][:,1] *= sign(ref_orientation[-1])
            this.radiated['P'][:,1] *= sign(ref_orientation[-1])

        # Store remant Quantities
        if verbose: alert('Using radiated quantity time series to calculate remnant quantity time series.')
        this.remnant = {}
        this.remnant['mask'] = mask
        this.remnant['time_used'] = this.radiated['time_used']
        this.remnant['M'] = 1 - this.radiated['E']
        this.remnant['Mw'] = this.remnant['M'] * this.lm[2,2]['psi4'].dphi[ mask ]/2

        # Calculate the internal angular momentum by using either the final or initial angular momentum values
        this.remnant['J'] = -this.radiated['J']

        # # Use the final spin value to set the integration constant
        # this.remnant['J'] = this.remnant['J'] - this.remnant['J'][end_index,:] + this.Sf # Enforce consistency with final spin vector

        # Use the initial J value to set the integration constant
        initial_index = 0 # index location where we expect the simulation data to NATURALLY start. For example, we expect that if the waveform data has been padded upon loading that this padding happens to the right of the data series, and not to the left.
        this.remnant['J'] = this.remnant['J'] - this.remnant['J'][initial_index,:] + this.J # Enforce consistency with initial spin vector

        this.remnant['S'] = this.remnant['J'] # The remnant has no orbital angular momentum. Is this right?
        this.remnant['P'] = -this.radiated['P'] # Assumes zero linear momentum at integration region start
        this.remnant['X'] = vstack([ this.remnant['J'][:,k]/(this.remnant['M']**2) for k in range(3) ]).T
        #
        if verbose: alert('Radiated quantities are stored to "this.radiated", and remnant quantities are stored to "this.remnant". Both are dictionaries.')
        return None

    #
    def __calc_radiated_linear_momentum__(this,mask):
        ''' Reference: https://arxiv.org/pdf/0707.4654.pdf '''

        # Import usefuls
        from numpy import sqrt,pi,vstack,zeros_like,array

        # NOTE:
        # * Extraction radii are already scaled in by default
        # * The paper uses a strange cnvetion for Imag which we account for here
        # * The user should be wary of tetrad differnces.

        # Pre-allocate arrays
        dPp,dPz = zeros_like(this.t,dtype=complex),zeros_like(this.t,dtype=complex)
        nothing = zeros_like(this.t,dtype=complex)
        # Define intermediate functions for coeffs
        a = lambda l,m: sqrt( (l-m)*(l+m+1.0) ) / ( l*(l+1) )
        b = lambda l,m: (1.0/(2*l)) * sqrt( ( (l-2)*(l+2)*(l+m)*(l+m-1.0) )/( (2*l-1)*(2*l+1) ) )
        c = lambda l,m: m*2.0 / ( l*(l+1) )
        d = lambda l,m: (1.0/l) * sqrt( (l-2)*(l+2)*(l-m)*(l+float(m))/( (2*l-1)*(2*l+1) ) )

        # Define shorthand for accessing strain and news
        f  = lambda l,m: this.lm[l,m]['news'].y   if (l,m) in this.lm else nothing

        # Sum over l,m ; NOTE that the overall scale factors will be applied later
        for l,m in this.lm:
            # Eq. 3.14
            dPp += f(l,m) * ( a(l,m)*f(l,m+1).conj() + b(l,-m)*f(l-1,m+1).conj() - b(l+1,m+1)*f(l+1,m+1).conj() )
            # Eq. 3.15
            dPz += f(l,m) * ( c(l,m)*f(l,m).conj() * d(l,m)*f(l-1,m).conj() + d(l+1,m)*f(l+1,m).conj() )

        # Apply overall operations
        dPp =  dPp.imag / ( 8*pi )
        dPz =  dPz.imag / ( 16*pi )

        # Unpack in-place linear momentum rate
        dPx,dPy = dPp.real,dPp.imag

        # Integrate to get angular momentum
        Px = spline_antidiff( this.t[mask],dPx[mask],k=3 )
        Py = spline_antidiff( this.t[mask],dPy[mask],k=3 )
        Pz = spline_antidiff( this.t[mask],dPz[mask],k=3 )

        #
        ans = vstack([Px,Py,Pz]).T
        return ans

    #
    def __calc_radiated_angular_momentum__(this,mask):
        ''' Reference: https://arxiv.org/pdf/0707.4654.pdf '''

        # Import usefuls
        from numpy import sqrt,pi,vstack,zeros_like,array

        # NOTE:
        # * Extraction radii are already scaled in by default
        # * The paper uses a strange cnvetion for Imag which we account for here
        # * The user should be wary of tetrad differnces.

        #
        dJx,dJy,dJz = zeros_like(this.t,dtype=complex), zeros_like(this.t,dtype=complex), zeros_like(this.t,dtype=complex)
        nothing = zeros_like(this.t,dtype=complex)
        F = lambda l,m: sqrt( l*(l+1) - m*(m+1) )

        # Sum over l,m ; NOTE that the overall scale factors will be applied later
        for l,m in this.lm:
            #
            hlm  = this.lm[l,m]['strain'].y
            flm  = this.lm[l,m]['news'].y
            fmp1 = this.lm[l,m+1]['news'].y if (l,m+1) in this.lm else nothing
            fmm1 = this.lm[l,m-1]['news'].y if (l,m-1) in this.lm else nothing
            # Eq. 3.22
            dJx += hlm * ( F(l,m)*fmp1.conj() + F(l,-m)*fmm1.conj() )
            # Eq. 3.23
            dJy += hlm * ( F(l,m)*fmp1.conj() - F(l,-m)*fmm1.conj() )
            # Eq. 3.24
            dJz += hlm * m * flm.conj()

        # Apply overall operations
        dJx = dJx.imag / ( 32*pi )
        dJy = dJy.real / ( 32*pi )
        dJz = dJz.imag / ( 16*pi )

        # Integrate to get angular momentum
        Jx = spline_antidiff( this.t[mask],dJx[mask],k=3 )
        Jy = spline_antidiff( this.t[mask],dJy[mask],k=3 )
        Jz = spline_antidiff( this.t[mask],dJz[mask],k=3 )

        #
        ans = -vstack([Jx,Jy,Jz]).T
        return ans

    # Create hybrid multipoles
    def hybridize( this, pn_w_orb_min=None, pn_w_orb_max=None, verbose=True, plot=None, aggressive=1 ):
        '''
        Create hybrid multipoles: This is effectively a wrapper for the make_pnnr_hybrid workflow.
        '''
        # Import hybrid class from nrutils
        from nrutils.manipulate.hybridize import make_pnnr_hybrid

        #
        w_orb_min = this.wstart_pn/2
        w_orb_merger = this[2,2]['psi4'].dphi[ this[2,2]['psi4'].k_amp_max ]/2

        #
        if pn_w_orb_min is None: pn_w_orb_min = 0.8*w_orb_min
        if pn_w_orb_max is None: pn_w_orb_max = (4*w_orb_merger+6*w_orb_min)/10

        # Initiate class instance
        hybo = make_pnnr_hybrid( this,                      # gwylm obj
                                 pn_w_orb_min=pn_w_orb_min, # start of PN freq
                                 pn_w_orb_max=pn_w_orb_max, #   end of PN freq
                                 kind = 'psi4',
                                 plot = plot,
                                 aggressive=aggressive,     # 2: force alignment of multipole phases
                                 verbose=verbose)
        # Return hyrbid object as well as related hybridized gwylm object
        return hybo.hybrid_gwylmo, hybo

    # Los pass filter using romline in basics.py to determine window region
    def lowpass(this):

        #
        msg = 'Howdy, partner! This function is experimental and should NOT be used.'
        error(msg,'lowpass')

        #
        from numpy import log,ones
        from matplotlib.pyplot import plot,show,axvline

        #
        for y in this.ylm:
            N = 8
            if y.m>=0:
                mask = y.f>0
                lf = log( y.f[ mask  ] )
                lamp = log( y.fd_amp[ mask  ] )
                knots,_ = romline(lf,lamp,N,positive=True,verbose=True)
                a,b = 0,1
                state = knots[[a,b]]
                window = ones( y.f.shape )
                window[ mask ] = maketaper( lf, state )
            elif y.m<0:
                mask = y.f<=0
                lf = log( y.f[ mask  ] )
                lamp = log( y.fd_amp[ mask  ] )
                knots,_ = romline(lf,lamp,N,positive=True,verbose=True)
                a,b = -1,-2
                state = knots[[a,b]]
                window = ones( y.f.shape )
                window[ mask ] = maketaper( lf, state )
            plot( lf, lamp )
            plot( lf, log(window[mask])+lamp, 'k', alpha=0.5 )
            plot( lf[knots], lamp[knots], 'o', mfc='none', ms=12 )
            axvline(x=lf[knots[a]],color='r')
            axvline(x=lf[knots[b]],color='r')
            # show()
            # plot(y.f,y.fd_amp)
            # show()
            plot( window )
            axvline(x=knots[a],color='r')
            axvline(x=knots[b],color='r')
            # show()
            y.fdfilter( window )

        #
        this.__lowpassfiltered__ = True



# Time Domain LALSimulation Waveform Approximant h_plus and cross, but using nrutils data conventions
def lswfa( apx      ='IMRPhenomPv2',    # Approximant name; must be compatible with lal convenions
           eta      = None,           # symmetric mass ratio
           chi1     = None,           # spin1 iterable (Dimensionless)
           chi2     = None,           # spin2 iterable (Dimensionless)
           fmin_hz  = 30.0,           # phys starting freq in Hz
           verbose  = False ):        # boolean toggle for verbosity

    #
    from numpy import array,linspace,double
    import lalsimulation as lalsim
    from nrutils import eta2q
    import lal

    # Standardize input mass ratio and convert to component masses
    M = 70.0
    q = eta2q(eta)
    q = double(q)
    q = max( [q,1.0/q] )
    m2 = M * 1.0 / (1.0+q)
    m1 = float(q) * m2

    # NOTE IS THIS CORRECT????
    S1 = array(chi1)
    S2 = array(chi2)

    #
    fmin_phys = fmin_hz
    M_total_phys = (m1+m2) * lal.MSUN_SI

    #
    TD_arguments = {'phiRef': 0.0,
             'deltaT': 1.0 * M_total_phys * lal.MTSUN_SI / lal.MSUN_SI,
             'f_min': fmin_phys,
             'm1': m1 * lal.MSUN_SI,
             'm2' : m2 * lal.MSUN_SI,
             'S1x' : S1[0],
             'S1y' : S1[1],
             'S1z' : S1[2],
             'S2x' : S2[0],
             'S2y' : S2[1],
             'S2z' : S2[2],
             'f_ref': 100.0,
             'r': lal.PC_SI,
             'z': 0,
             'i': 0,
             'lambda1': 0,
             'lambda2': 0,
             'waveFlags': None,
             'nonGRparams': None,
             'amplitudeO': -1,
             'phaseO': -1,
             'approximant': lalsim.SimInspiralGetApproximantFromString(apx)}

    #

    # Use lalsimulation to calculate plus and cross in lslsim dataformat
    hp, hc  = lalsim.SimInspiralTD(**TD_arguments)

    # Convert the lal datatype to a gwf object
    D = 1e-6 * TD_arguments['r']/lal.PC_SI
    y = lalsim2gwf( hp,hc,m1+m2, D )

    #
    return y




# Frequency Domain Domain LALSimulation Waveform Approximant h_plus and cross, but using nrutils data conventions
def lswfafd( apx      ='IMRPhenomPv2',    # Approximant name; must be compatible with lal convenions
           eta      = None,           # symmetric mass ratio
           chi1     = None,           # spin1 iterable (Dimensionless)
           chi2     = None,           # spin2 iterable (Dimensionless)
           fmin_hz  = 30.0,           # phys starting freq in Hz
           verbose  = False ):        # boolean toggle for verbosity

    #
    from nrutils.core.units import codef
    from numpy import array,linspace,double
    import lalsimulation as lalsim
    from nrutils import eta2q
    import lal

    #
    error('This function is in dev/incomplete.')

    # Standardize input mass ratio and convert to component masses
    M = 70.0
    q = eta2q(eta)
    q = double(q)
    q = max( [q,1.0/q] )
    m2 = M * 1.0 / (1.0+q)
    m1 = float(q) * m2

    #
    df_code = 0.01
    fmax_code = 0.5


    # NOTE IS THIS CORRECT????
    S1 = array(chi1)
    S2 = array(chi2)

    #
    fmin_phys = fmin_hz
    M_total_phys = (m1+m2) * lal.MSUN_SI

    #
    # REAL8 phiRef, REAL8 deltaF, REAL8 m1, REAL8 m2, REAL8 S1x, REAL8 S1y, REAL8 S1z, REAL8 S2x, REAL8 S2y, REAL8 S2z, REAL8 f_min, REAL8 f_max, REAL8 f_ref, REAL8 r, REAL8 z, REAL8 i, REAL8 lambda1, REAL8 lambda2, SimInspiralWaveformFlags waveFlags, SimInspiralTestGRParam nonGRparams, int amplitudeO, int phaseO, Approximant approximant

    #
    FD_arguments = {'phiRef': 0.0,
             'deltaF': codef(df_code,M),
             'f_min': fmin_phys,
             'm1': m1 * lal.MSUN_SI,
             'm2' : m2 * lal.MSUN_SI,
             'S1x' : S1[0],
             'S1y' : S1[1],
             'S1z' : S1[2],
             'S2x' : S2[0],
             'S2y' : S2[1],
             'S2z' : S2[2],
             'f_ref': 100.0,
             'f_max': 200.0,
             'r': lal.PC_SI,
             'z': 0,
             'i': 0,
             'lambda1': 0,
             'lambda2': 0,
             'waveFlags': None,
             'nonGRparams': None,
             'amplitudeO': -1,
             'phaseO': -1,
             'approximant': lalsim.SimInspiralGetApproximantFromString(apx)}

    #

    # Use lalsimulation to calculate plus and cross in lslsim dataformat
    hp, hc  = lalsim.SimInspiralFD(**FD_arguments)

    # Convert the lal datatype to a gwf object
    D = 1e-6 * FD_arguments['r']/lal.PC_SI
    #y = lalsim2gwf( hp,hc,m1+m2, D, fd=True )
    y = (hp,hc)

    #
    return y




# Characterize END of time domain waveform (POST RINGDOWN)
class gwfcharend:

    def __init__(this,ylm):
        # Use amplitude
        this.__characterize_amplitude__(ylm)
        # Use frequency
        this.__characterize_frequency__(ylm)

    # Characterize the end of the waveform using values of the amplitude
    def __characterize_amplitude__(this,ylm):
        # Import useful things
        from numpy import log
        # ROM (Ruduce order model) the post-peak as two lines
        amp = ylm.amp[ ylm.k_amp_max: ]
        t = ylm.t[ ylm.k_amp_max: ]
        mask = amp > 0
        amp = amp[mask]
        t = t[mask]
        la = log( amp )
        tt = t
        # ax,fig = ylm.plot()
        # from matplotlib.pyplot import axvline,sca,show
        # sca(ax[0]); axvline( ylm.t[ylm.k_amp_max] )
        # print '>> ',tt.shape, la.shape
        # show()
        knots,rl = romline(tt,la,2)
        # Check for lack of noise floor (in the case of sims stopped before noise floor reached)
        # NOTE that in this case no effective windowing is applied
        this.nonoisefloor = knots[-1]+1 == len(tt)
        if this.nonoisefloor:
            msg = 'No noise floor found. This simulation may have been stopped before the numerical noise floor was reached.'
            warning(msg,'gwfcharend')
        # Define the start and end of the region to be windowed
        this.left_index = ylm.k_amp_max + knots[-1]
        this.right_index = ylm.k_amp_max + knots[-1]+(len(tt)-knots[-1])*6/10
        # Calculate the window and store to the current object
        this.window_state = [ this.right_index, this.left_index ]
        this.window = maketaper( ylm.t, this.window_state )

    # Characterize the end of the waveform using values of the frequency
    def __characterize_frequency__(this,ylm):
        # Import usefuls
        from numpy import argmax,diff
        #
        B = ylm.t > ylm.t[ ylm.k_amp_max ]
        #
        a = upbow(ylm.dphi[B])
        knots,rl = romline(ylm.t[B],a,5)
        #
        this.frequency_end_index = ylm.k_amp_max + ( knots[0] if knots[0]>0 else knots[1] )
        #
        return None


# Characterize the START of a time domain waveform (PRE INSPIRAL)
class gwfcharstart:

    #
    def __init__( this,                 # the object to be created
                  y,                    # input gwf object who'se start behavior will be characterised
                  shift     = 3,        # The size of the turn on region in units of waveform cycles.
                  __smooth__ = True,
                  verbose   = False ):

        #
        from numpy import arange,diff,where,array,ceil,mean,ones_like,argmax
        from numpy import histogram as hist
        thisfun=this.__class__.__name__

        # Take notes on what happens
        notes = []

        # This algorithm estimates the start of the gravitational waveform -- after the initial junk radiation that is present within most raw NR output. The algorithm proceeds in the manner consistent with a time domain waveform.

        # Validate inputs
        if not (y.__class__.__name__=='gwf'):
            msg = 'First imput must be a '+cyan('gwf')+' object. Type %s found instead.' % y.__class__.__name__
            error(msg,thisfun)


        # 1. Find the pre-peak portion of the waveform.
        val_mask = arange( y.k_amp_max )
        # 2. Find the peak locations of the plus part. NOTE that smooth() is defined in positive.maths
        pks,pk_mask = findpeaks( smooth( y.cross[ val_mask ], 20 ).answer if __smooth__ else y.cross[ val_mask ] )
        # pks,pk_mask = findpeaks( y.cross[ val_mask ] )
        pk_mask = pk_mask[ pks > y.amp[y.k_amp_max]*5e-4 ]

        # 3. Find the difference between the peaks
        D = diff(pk_mask)

        # If the waveform starts at its peak (e.g. in the case of ringdown)
        if len(D)==0:

            #
            this.left_index = 0
            this.right_index = 0
            this.left_dphi=this.center_dphi=this.right_dphi = y.dphi[this.right_index]
            this.peak_mask = [0]

        else:

            # 4. Find location of the first peak that is separated from its adjacent by greater/equal than the largest value. This location is stored to start_map.
            start_map = find(  D >= max(D)  )[0]

            # 5. Determine the with of waveform turn on in indeces based on the results above. NOTE that the width is bound below by half the difference betwen the wf start and the wf peak locations.
            safedex = min( len(pk_mask)-1, start_map+shift )
            index_width = min( [ 1+pk_mask[safedex]-pk_mask[start_map], 0.5*(1+y.k_amp_max-pk_mask[ start_map ]) ] )
            # 6. Estimate where the waveform begins to turn on. This is approximately where the junk radiation ends. Note that this area will be very depressed upon windowing, so is can be
            (_,j_id) = find_amp_peak_index( y.t, y.amp, return_jid=True )
            # NOTE that the line above is more robust for precessing cases than the line below. There are cases when the optimal emission axis crosses z=0 which causes problems for the line below
            # j_id = pk_mask[ start_map ]

            # 7. Use all results thus far to construct this object
            this.left_index     = int(j_id)                                         # Where the initial junk radiation is thought to end
            this.right_index    = int(j_id + index_width - 1)                       # If tapering is desired, then this index will be
                                                                                    # the end of the tapered region.
            this.left_dphi      = y.dphi[ this.left_index  ]                        # A lowerbound estimate for the min frequency within
                                                                                    # the waveform.
            this.right_dphi     = y.dphi[ this.right_index ]                        # An upperbound estimate for the min frequency within
                                                                                    # the waveform
            this.center_dphi    = mean(y.dphi[ this.left_index:this.right_index ])  # A moderate estimate for the min frequency within they
                                                                                    # waveform
            this.peak_mask      = pk_mask

        # Construct related window
        this.window_state = [this.left_index,this.right_index]
        this.window = maketaper( y.t, this.window_state )


# Function which converts lalsim waveform to gwf object
def lalsim2gwf( hp,hc,M,D, fd=False ):

    #
    from numpy import linspace,array,double,sqrt,hstack,zeros
    from nrutils.tools.unit.conversion import codeh

    # Extract plus and cross data. Divide out contribution from spherical harmonic towards NR scaling
    x = sYlm(-2,2,2,0,0)
    h_plus  = hp.data.data/x
    h_cross = hc.data.data/x

    if not fd:
        # Create time series data
        t = linspace( 0.0, (h_plus.size-1.0)*hp.deltaT, int(h_plus.size) )
        # Create waveform
        harr = array( [t,h_plus,h_cross] ).T
        # Convert to code units, where Mtotal=1
        harr = codeh( harr,M,D )
        # Create gwf object
        h = gwf( harr, kind=r'$h^{\mathrm{lal}}_{22}$' )
    else:
        # Create time series data
        f = linspace( 0.0, (h_plus.size-1.0)*hp.deltaT, int(h_plus.size) )
        # Create waveform
        harr = array( [t,h_plus,h_cross] ).T
        # Convert to code units, where Mtotal=1
        harr = codehf( harr,M,D )

    #
    return h


# Taper a waveform object
def gwftaper( y,                        # gwf object to be windowed
              state,                    # Index values defining region to be tapered:
                                        # For state=[a,b], if a>b then the taper is 1 at b and 0 at a
                                        # if a<b, then the taper is 1 at a and 0 at b.
              plot      = False,
              verbose   = False):

    # Import useful things
    from numpy import ones
    from numpy import hanning as hann

    # Parse taper state
    a = state[0]
    b = state[-1]

    # Only proceed if a valid window is given
    proceed = True
    true_width = abs(b-a)
    twice_hann = hann( 2*true_width )
    if b>a:
        true_hann = twice_hann[ :true_width ]
    elif a<b:
        true_hann = twice_hann[ true_width: ]
    else:
        proceed = False

    # Proceed (or not) with windowing
    window = ones( y.n )
    if proceed:
        # Make the window
        window[ :min(state) ] = 0
        window[ min(state) : max(state) ] = true_hann
        # Apply the window to the data and reset fields
        y.wfarr[:,1] *= window
        y.wfarr[:,2] *= window
        y.setfields()

    #
    return window



# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Given the location of an lvcnr h5 file, as well as the
# desired multipoles to load, generate a gwylm object.
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
def lvcnr5_to_gwylm(h5dir,lm=None,verbose=True,dt=0.25,lmax=6,clean=True):
    '''
    Given the location of an lvcnr h5 file, as well as the desired multipoles to load,
    generate a gwylm object.
    '''

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.stats.mstats import mode
    from scipy.version import version as scipy_version
    from numpy import array,arange,exp,inf,sqrt, nan,sign,amax
    from nrutils import scentry,gwylm,gwf

    import h5py

    #
    lmlist = lm

    #
    f = h5py.File( h5dir )

    #
    e = scentry( None, None )
    e.m1,e.m2 = f.attrs['mass1'], f.attrs['mass2']
    chi1 = array([f.attrs['spin1x'],f.attrs['spin1y'],f.attrs['spin1z']])
    chi2 = array([f.attrs['spin2x'],f.attrs['spin2y'],f.attrs['spin2z']])
    e.X1,e.X2 = chi1,chi2
    e.S1,e.S2 = chi1*e.m1**2,chi2*e.m1**2

    if f.attrs['Format'] == 2:
        e.R1 = array( [f['position1x-vs-time']['Y'][0],f['position1y-vs-time']['Y'][0],f['position1z-vs-time']['Y'][0]] )
        e.R2 = array( [f['position2x-vs-time']['Y'][0],f['position2y-vs-time']['Y'][0],f['position2z-vs-time']['Y'][0]] )
        R = e.R2-e.R1
        e.b = sqrt(sum( R*R ))
    else:
        e.R1 = nan
        e.R2 = nan
        R = nan
        e.b = nan

    if f.attrs['Format'] == 3:
        Xf = array( [f['remnant-spinx-vs-time']['Y'][-1],f['remnant-spiny-vs-time']['Y'][-1],f['remnant-spinz-vs-time']['Y'][-1]] )
        e.xf = sqrt( sum( Xf*Xf ) )
        e.mf = f['remnant-mass-vs-time']['Y'][-1]
    else:
        e.mf = nan
        Xf = nan
        e.xf = nan


    e.default_extraction_par = inf
    e.default_level = None
    e.config = None
    e.simname = h5dir.split('/')[-1].split('.')[0]
    e.setname = f.attrs['NR-group']+'-'+f.attrs['type']
    e.label = 'unknown-label'
    e.eta = e.m1*e.m2 / ( (e.m1+e.m2)**2 )

    #
    y = gwylm(e,load=False)

    try:
        nrtimes = f['NRtimes']
    except:
        nrtimes = f['amp_l2_m2']['X'][:]
    t = arange( min(nrtimes),max(nrtimes)+dt,dt )
    #
    done = False

    # Generate l,m values based on lmax
    if lmlist is None: lmlist = [ (l,m)  for l in range(2,lmax+1) for m in range(-l,l+1) ]

    #
    for l,m in lmlist:
        if verbose: alert('Loading strain for %s'%cyan('(l,m) = (%i,%i)'%(l,m)))
        try:
            amp = spline(f['amp_l%i_m%i'%(l,m)]['X'],f['amp_l%i_m%i'%(l,m)]['Y'])(t)
            pha = spline(f['phase_l%i_m%i'%(l,m)]['X'],f['phase_l%i_m%i'%(l,m)]['Y'])(t)
        except:
            if verbose: alert("couldn't load (l,m)=({0},{1})".format(l,m))
            break

        # Enforce internal sign convention for time domain hlm
        msk_ = amp > 0.01*amax(amp)
        if int(scipy_version.split('.')[1])<16:
            # Account for old scipy functionality
            external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( pha ) )[0][0]
        else:
            # Account for modern scipy functionality
            external_sign_convention = sign(this.L[-1]) * sign(m) * mode( sign( pha ) ).mode[0]
        if M_RELATIVE_SIGN_CONVENTION != external_sign_convention:
            pha = -pha
            msg = yellow('Re-orienting waveform phase')+' to be consistent with internal sign convention for Strain, where sign(dPhi/dt)=%i*sign(m).' % M_RELATIVE_SIGN_CONVENTION + ' Note that the internal sign convention is defined in ... nrutils/core/__init__.py as "M_RELATIVE_SIGN_CONVENTION". This message has appeared becuase the waveform is determioned to obey and sign convention: sign(dPhi/dt)=%i*sign(m).'%(external_sign_convention)
            thisfun=inspect.stack()[0][3]
            if verbose: alert( msg )

        z = amp * exp(1j*pha)
        wfarr = array([ t, z.real, z.imag ]).T
        y.hlm.append(  gwf( wfarr,l=l,m=m,kind='$rh_{%i%i}/M$'%(l,m) )  )

    #
    y.__lmlist__ = lmlist
    y.__input_lmlist__ = lmlist
    y.__curate__()
    f.close()

    #
    y.characterize_start_end()
    if clean: y.clean()

    #
    return y
