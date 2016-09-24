
'''
Modules for Numerical Relativity Simulation Catalog:

* catalog: builds catalog given a cinfiguration file, or directory containing many configuration files.

* scentry: class for simulation catalog entry (should include io)

'''

#
from nrutils.core import settings as gconfig
from nrutils.core.basics import *
from nrutils.core import M_RELATIVE_SIGN_CONVENTION
import warnings,sys

# Class representation of configuration files. The contents of these files define where the metadata for each simulation is stored, and where the related NR data is stored.
class scconfig(smart_object):

    # Create scconfig object from configuration file location
    def __init__(this,config_file_location=None,overwrite=True):

        # Required fields from smart_object
        this.source_file_path = []
        this.source_dir = []
        this.overwrite = overwrite

        # call wrapper for constructor
        this.config_file_location = config_file_location
        this.reconfig()

    # The actual constructor: this will be called within utility functions so that scentry objects are configured with local settings.
    def reconfig(this):

        #
        if this.config_file_location is None:
            msg = '(!!) scconfig objects cannot be initialted/reconfigured without a defined "config_file_location" location property (i.e. string where the related config file lives)'
            raise ValueError(msg)

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

        # In some cases, it is useful to have this function return this
        return this

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
                           'handler_location',          # location of python script which contains validator and
                                                        # learn_metadata functions
                           'is_extrapolated',           # users should set this to true if waveform is extrapolated
                                                        # to infinity
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
        if not isinstance( this.catalog_dir, str ):
            msg = 'catalog_dir values must be string'
            error(red(msg),thisfun)


        if 2 != len(this.default_par_list):
            msg = '(!!) Error in %s: default_par_list must be list containing default extraction parameter (Numeric value) and default level (also Numeric in value). Invalide case found: %s' % (magenta(this.config_file_location),list(this.default_par_list))
            raise ValueError(msg)

        # Make sure that all directories end with a forward slash
        for attr in this.__dict__:
            if 'dir' in attr:
                if this.__dict__[attr][-1] != '/':
                    this.__dict__[attr] += '/'
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
    def __init__( this, config_obj, metadata_file_location ):

        # Keep an internal log for each scentry created
        this.log = '[Log for %s] The file is "%s".' % (this,metadata_file_location)

        # Store primary inputs as object attributes
        this.config = config_obj
        this.metadata_file_location = metadata_file_location

        # Validate the location of the metadata file: does it contain waveform information? is the file empty? etc
        this.isvalid = this.validate()

        # If valid, learn metadata. Note that metadata property are defined as none otherise. Also NOTE that the standard metadata is stored directly to this object's attributes.
        this.raw_metadata = None
        if this.isvalid is True:
            #
            print '## Working: %s' % cyan(metadata_file_location)
            this.log += ' This entry\'s metadata file is valid.'

            # i.e. learn the meta_data_file
            # this.learn_metadata()
            # this.label = sclabel( this )
            try:
                this.learn_metadata()
                this.label = sclabel( this )
            except:
                emsg = sys.exc_info()[1].message
                this.log += '%80s'%' [FATALERROR-1] The metadata failed to be read. There may be an external formatting inconsistency. It is being marked as invalid with None. The system says: %s'%emsg
                warning( 'The following error message will be logged: '+red(emsg),'scentry')
                this.isvalid = None # An external program may use this to do something
                this.label = 'invalid!'

        elif this.isvalid is False:
            print '## The following is '+red('invalid')+': %s' % cyan(metadata_file_location)
            this.log += ' This entry\'s metadta file is invalid.'

    #
    def validate(this):

        # determine the validity of the current object

        # import validation function given in config file
        from imp import load_source
        validator_module = load_source( '', this.config.handler_location )

        # Name the function representation that will be used to load the metadata file, and convert it to raw and standardized metadata
        validator = validator_module.validate

        # vet the directory where the metadata file lives for: waveform and additional metadata
        status = validator( this.metadata_file_location, config = this.config )

        #
        return status

    # Standardize metadata
    def learn_metadata(this):

        #
        from imp import load_source
        from numpy import allclose

        # import metadata converter functiongiven in config file
        converter_module = load_source( '', this.config.handler_location )

        # Name the function representation that will be used to load the metadata file, and convert it to raw and standardized metadata
        learn_institute_metadata = converter_module.learn_metadata

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
                           'extraction_parameter_is_radius', # If False, then raw data must be scaled by the extraction
                                                             # radius (it's ~ 1 ); else, raw data is not scaled by the
                                                             # extractino radius (it's ~ 1/r, and it will need to be
                                                             # scaled by r to then be proportional to 1).
                           'xf' ]           # Final dimensionless spin *Magnitude*: xf = |Sf|/(mf*mf)

        for attr in required_attrs:
            if attr not in standard_metadata.__dict__:
                msg = '(!!) Error -- Output of %s does NOT contain required field %s' % ( this.config.handler_location, attr )
                raise ValueError(msg)

        # Confer the required attributes to this object for ease of referencing
        for attr in standard_metadata.__dict__.keys():
            setattr( this, attr, standard_metadata.__dict__[attr] )

        # tag this entry with its inferred setname
        this.setname = this.raw_metadata.source_dir[-1].split( this.config.catalog_dir )[-1].split('/')[0]

        # tag this entry with the directory location of the metadata file. NOTE that the waveform data must be reference relative to this directory via config.data_file_name_format
        this.relative_simdir = this.raw_metadata.source_dir[-1].split( this.config.catalog_dir )[-1]

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
        # Enfore m1>m2 convention.
        satisfies_massratio_convetion = lambda e: (not e.m1 > e.m2) and (not allclose(e.m1,e.m2,atol=1e-4))
        if satisfies_massratio_convetion(this):
            this.flip()
        if satisfies_massratio_convetion(this):
            msg = 'Mass ratio convention m1>m2 must be used. Check scentry.flip(). It should have corrected this! \n>> m1 = %g, m2 = %g' % (this.m1,this.m2)
            raise ValueError(msg)

    # Create dynamic function that references the user's current configuration to construct the simulation directory of this run.
    def simdir(this):
        ans = this.config.reconfig().catalog_dir + this.relative_simdir
        if not this.config.config_exists:
            msg = 'The current object has been marked as '+red('non-existent')+', likely by reconfig(). Please verify that the ini file for the related run exists. You may see this message for other (yet unpredicted) reasons.'
            error(msg,'scentry.simdir()')
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

        # Apply the flip to the current object
        this.R1 = R1; this.R2 = R2
        this.m1 = m1; this.m2 = m2
        this.P1 = P1; this.P2 = P2
        this.L1 = L1; this.L2 = L2
        this.S1 = S1; this.S2 = S2

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
    cpath_list = glob.glob( gconfig.config_path+'*.ini' )

    # If a keyword is give, filter against found config files
    if isinstance(keyword,(str,unicode)):
        msg = 'Filtering ini files for \"%s\"'%cyan(keyword)
        alert(msg,'scbuild')
        cpath_list = filter( lambda path: keyword in path, cpath_list )

    #
    if not cpath_list:
        msg = 'Cannot find configuration files (*.ini) in %s' % gconfig.config_path
        error(msg,thisfun)

    # Create config objects from list of config files
    configs = [ scconfig( config_path ) for config_path in cpath_list ]

    # For earch config
    for config in configs:

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        # Create streaming log file        #
        logfstr = gconfig.database_path + '/' + splitext(basename(config.config_file_location))[0] + '.log'
        msg = 'Opening log file in: '+cyan(logfstr)
        alert(msg,thisfun)
        logfid = open(logfstr, 'w')
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

        # Search recurssively within the config's catalog_dir for files matching the config's metadata_id
        msg = 'Searching for %s in %s.' % ( cyan(config.metadata_id), cyan(config.catalog_dir) ) + yellow(' This may take a long time if the folder being searched is mounted from a remote drive.')
        alert(msg,thisfun)
        mdfile_list = rfind(config.catalog_dir,config.metadata_id,verbose=True)
        alert('done.',thisfun)

        # (try to) Create a catalog entry for each valid metadata file
        catalog = []
        h = -1
        for mdfile in mdfile_list:

            # Create tempoary scentry object
            entry = scentry(config,mdfile)

            # Write to the master log file
            h+=1
            logfid.write( '%5i\t%s\n'% (h,entry.log) )

            # If the obj is valid, add it to the catalog list, else ignore
            if entry.isvalid:
                catalog.append( entry )
            else:
                del entry

        # Store the catalog to the database_path
        if save:
            db = gconfig.database_path + '/' + splitext(basename(config.config_file_location))[0] + '.' + gconfig.database_ext
            msg = 'Saving database file to %s'%cyan(db)
            alert(msg,'scbuild')
            with open(db, 'wb') as dbf:
                pickle.dump( catalog , dbf, pickle.HIGHEST_PROTOCOL )

        # Close the log file
        logfid.close()

        #
        wave_train = ''#'~~~~<vvvvvvvvvvvvvWw>~~~~'
        hline = wave_train*3
        msg = '\n\n#%s#\n%s with \"%s\". The related log file is at \"%s\".\n#%s#'%(hline,hlblack('Done'),green(config.catalog_dir),green(logfstr),hline)
        alert(msg,'scbuild')




# Function for searching through catalog files.
def scsearch( catalog = None,           # Manually input list of scentry objects to search through
              q = None,                 # RANGE of mass ratios (>=1) to search for
              nonspinning = None,       # Non-spinning initially
              spinaligned = None,       # spin-aligned with L AND no in-plane spin INITIALLY
              spinantialigned = None,   # spin-anti-aligned with L AND no in-plane spin INITIALLY
              precessing = None,        # not spin aligned
              notprecessing = None,     # not precessing
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
              verbose = None):          # be verbose

    # Print non None inputs to screen
    thisfun = inspect.stack()[0][3]
    if verbose is not None:
        for k in dir():
            if (eval(k) is not None) and (k != 'thisfun'):
                print '[%s]>> Found %s (=%r) keyword.' % (thisfun,textul(k),eval(k))

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
        dblist = glob.glob( gconfig.database_path+'*.'+gconfig.database_ext )
        # Load the catalog file(s)
        catalog = []
        for db in dblist:
            with open( db , 'rb') as dbf:
                catalog = catalog + pickle.load( dbf )

    # mass-ratio
    if q is not None:
        # handle int of float input
        if isinstance(q,(int,float)): q = [q-1e-6,q+1e-6]
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

    # not precessing, same as spinaligned & spin anti aligned
    if notprecessing is True:
        test = lambda k: allclose( abs(dot(k.S1+k.S2,k.L1+k.L2))/(norm(k.L1+k.L2)*norm(k.S1+k.S2)), 1.0 , atol = tol )
        catalog = filter( test, catalog )

    # spins have equal magnitude
    if equalspin is True:
        test = lambda k: allclose( norm(k.S1), norm(k.S2), atol = tol )
        catalog = filter( test, catalog )

    # spins have unequal magnitude
    if unequalspin is True:
        test = lambda k: not allclose( norm(k.S1), norm(k.S2), atol = tol )
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
            msg = '[%s]>> setname input must be nonempty string or list.' % thisfun
            raise ValueError(msg)

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
            msg = '[%s]>> notsetname input must be nonempty string or list.' % thisfun
            raise ValueError(msg)

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
            msg = '[%s]>> institute input must be nonempty string or list.' % thisfun
            raise ValueError(msg)

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
                alert(msg,'scsearch')
        else:
            allkeys = False # NOTE that this means: ANY keys will be passed
            if verbose:
                msg = 'List of keywords found: '+cyan('ANY scentry objects matching will be passed.')+' To pass ALL entries matching the keywords, input the kwywords using a list object.'
                alert(msg,'scsearch')

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
    if exists is not None:
        def isondisk(e):
            ans = (e.config).reconfig().config_exists and os.path.isdir(e.simdir())
            if not ans:
                msg = 'Ignoring entry at %s becuase its config file cannot be found and/or its simulation directory cannot be found.' % cyan(e.simdir())
                warning(msg,'scsearch')
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
    if verbose:
        if len(catalog)>0:
            print '## Found %s%s simulations:' % ( bold(str(len(catalog))), output_descriptor )
            for k,entry in enumerate(catalog):
                print '[%04i][%s] %s: %s' % ( k+1, green(entry.config.config_file_location.split('/')[-1].split('.')[0]), cyan(entry.setname), entry.label )
        else:
            print red('!! Found %s simulations.' % str(len(catalog)))
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

        # Spin on only 1st BH
        if allclose( norm(e.S1), 0.0, atol=tol ) and not allclose( norm(e.S2), 0.0, atol=tol ) :
            tag.append('s1')

        # Spin on only 2nd BH
        if allclose( norm(e.S2), 0.0, atol=tol ) and not allclose( norm(e.S1), 0.0, atol=tol ) :
            tag.append('s2')

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
                  l                     = None,
                  m                     = None,
                  extraction_parameter  = None,
                  kind                  = None, # strain or psi4
                  friend                = None, # gwf object from which to clone fields
                  verbose = False ):    # Verbosity toggle

        #
        this.dt = dt

        # The kind of obejct to be created : e.g. psi4 or strain
        if kind is None:
            kind = r'$y$'
        this.kind = kind

        # Set optional fields to none as default. These will be set externally is they are of use.
        this.l = l
        this.m = m
        this.extraction_parameter = extraction_parameter

        # use the raw waveform data to define all fields
        this.wfarr = wfarr
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

    # set fields of standard wf object
    def setfields(this,wfarr=None,dt=None):

        # If given dt, then interpolote waveform array accordingly
        if dt is not None:
            wfarr = intrp_wfarr(wfarr,delta=dt)

        # Alert the use if improper input is given
        if (wfarr is None) and (this.wfarr is None):
            msg = 'waveform array input (wfarr=) must be given'
            raise ValueError(msg)
        elif wfarr is not None:
            this.wfarr = wfarr

        ##########################################################
        # Make sure that waveform array is in t-plus-cross format #
        ##########################################################

        # Imports
        from numpy import abs,sign,linspace,exp,arange,angle,diff,ones
        from numpy import vstack,sqrt,unwrap,arctan,argmax,mod,floor
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
        this.valdt()
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
            this.valwfarr()

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
        k = find( (this.amp > 0) * (this.amp<max(this.amp)) )[0]
        phi_[0:k] = phi_[k]
        this.phi = phi_

        this.dphi   = intrp_diff( this.t, this.phi )                # Derivative of phase, last point interpolated to preserve length
        # this.dphi   = diff( this.phi )/this.dt                # Derivative of phase, last point interpolated to preserve length

        this.k_amp_max = argmax(this.amp)                           # index location of max ampitude
        this.intrp_t_amp_max = intrp_argmax(this.amp,domain=this.t) # Interpolated time coordinate of max

        #
        this.n      = len(this.t)                                   # Number of time samples
        this.window = ones( this.n )                                # initial state of time domain window
        this.fs     = 1.0/this.dt                                   # Sampling rate
        this.df     = this.fs/this.n                                # freq resolution

        # --------------------------------------------------- #
        # Always calculate frequency domain data
        # --------------------------------------------------- #

        # compute the frequency domain
        this.f = fftshift(fftfreq( this.n, this.dt ))

        # compute fourier transform values
        this.fd_plus   = fftshift(fft( this.plus  )) * this.dt                    # fft of plus
        this.fd_cross  = fftshift(fft( this.cross )) * this.dt                    # fft of cross
        this.fd_y       = this.fd_plus + 1j*this.fd_cross               # full fft
        this.fd_amp     = abs( this.fd_y )                              # amp of full fft
        this.fd_phi     = unwrap( angle( this.fd_y ) )                  # phase of full fft

        # this.fd_dphi    = diff( this.fd_phi )/this.df             # phase rate: dphi/df
        this.fd_dphi    = intrp_diff( this.f, this.fd_phi )             # phase rate: dphi/df

        this.fd_k_amp_max = argmax( this.fd_amp )

        # Starting frequency in rad/sec
        this.wstart = None

    # Copy attrributed from friend.
    def meet(this,friend,init=False,verbose=False):

        # If wrong type input, let the people know.
        if not isinstance(friend,gwf):
            msg = '1st input must be of type ' + bold(type(this).__name__)+'.'
            error( msg, fname=inspect.stack()[0][3] )

        # Copy attrributed from friend. If init, then do not check if attribute already exists in this.
        for attr in friend.__dict__:

            proceed = (attr in this.__dict__)
            proceed = proceed and type(friend.__dict__[attr]).__name__ in ('int','int64','float','scentry', 'string')

            # msg = '%s is %s and %s' % (attr,type(friend.__dict__[attr]).__name__,magenta('proceed=%r'%proceed))
            # alert(msg)

            if proceed or init:
                if verbose: print '\t that.%s --> this.%s (%s)' % (attr,attr,type(friend.__dict__[attr]).__name__)
                setattr( this, attr, friend.__dict__[attr] )

        #
        dir(this)
        return this

    # validate whether there is a constant time step
    def valdt(this):
        #
        from numpy import diff,var,allclose,vstack,mean,linspace,arange,array,double
        from scipy.interpolate import InterpolatedUnivariateSpline
        # note the shape convention
        t = this.wfarr[:,0]
        # if there is a non-uniform timestep, or if the input dt is not None and not equal to the given dt
        NONUNIFORMT = not isunispaced(t)
        INPUTDTNOTGIVENDT = this.dt is None
        if NONUNIFORMT:
            msg = '(**) Waveform not uniform in time-step. Interpolation will be applied.'
            print magenta(msg)
        if NONUNIFORMT and INPUTDTNOTGIVENDT:
            # if dt is not defined and not none, assume smallest dt
            if this.dt is None:
                this.dt = diff(lim(t))/len(t)
                msg = '(**) Warning: No dt given to gwf(). We will assume that the input waveform array is in geometric units, and that dt = %g will more than suffice.' % this.dt
                print magenta(msg)
            # interpolate waveform array
            intrp_t = this.dt * arange( 0, double(round( (t[-1]-t[0]))/this.dt )+1 ) + t[0]
            intrp_R = InterpolatedUnivariateSpline( t, this.wfarr[:,1] )( intrp_t )
            intrp_I = InterpolatedUnivariateSpline( t, this.wfarr[:,2] )( intrp_t )
            # create final waveform array
            this.wfarr = vstack([intrp_t,intrp_R,intrp_I]).T
        else:
            # otherwise, set dt automatically
            this.dt = mean(diff(t))

    # validate shape of waveform array
    def valwfarr(this):
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
              domain = None):

        # Handle which default domain to plot
        if domain is None:
            domain = 'time'
        elif not ( domain in ['time','freq'] ):
            msg = 'Error: domain keyword must be either "%s" or "%s".' % (cyan('time'),cyan('freq'))
            error(msg,'gwylm.plot')

        # Plot selected domain.
        if domain == 'time':
            ax = this.plottd( show=show,fig=fig,title=title )
        elif domain == 'freq':
            ax = this.plotfd( show=show,fig=fig,title=title )

        #
        return ax

    # Plot frequency domain
    def plotfd( this,
                show    =   False,
                fig     =   None,
                title   =   None,
                verbose =   False ):

        #
        from matplotlib.pyplot import plot,subplot,figure,tick_params,subplots_adjust
        from matplotlib.pyplot import grid,setp,tight_layout,margins,xlabel
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import ylabel as yl
        from matplotlib.pyplot import title as ttl
        from numpy import ones,sqrt,hstack,array

        #
        if fig is None:
            fig = figure(figsize = 1.1*array([8,7.2]))
            fig.set_facecolor("white")

        #
        kind = this.kind

        #
        clr = rgb(3)
        grey = 0.9*ones(3)
        lwid = 1
        txclr = 'k'
        fs = 18
        font_family = 'serif'

        #
        ax = []
        # xlim = lim(this.t) # [-400,this.t[-1]]

        #
        pos_mask = this.f>0


        # ------------------------------------------------------------------- #
        # Amplitude
        # ------------------------------------------------------------------- #
        ax.append( subplot(3,1,1) );
        grid(color='0.95', linestyle='-')
        setp(ax[-1].get_xticklabels(), visible=False)
        ax[-1].set_xscale('log', nonposx='clip')
        ax[-1].set_yscale('log', nonposy='clip')
        #
        plot( this.f[pos_mask], this.fd_amp[pos_mask], color=clr[0] )
        pylim( this.f[pos_mask], this.fd_amp[pos_mask], pad_y=10 )

        # ------------------------------------------------------------------- #
        # Total Phase
        # ------------------------------------------------------------------- #
        ax.append( subplot(3,1,2, sharex=ax[0]) );
        grid(color='0.95', linestyle='-')
        setp(ax[-1].get_xticklabels(), visible=False)
        ax[-1].set_xscale('log', nonposx='clip')
        #
        plot( this.f[pos_mask], this.fd_phi[pos_mask], color=1-clr[0] )
        pylim( this.f[pos_mask], this.fd_phi[pos_mask] )

        # ------------------------------------------------------------------- #
        # Total Phase Rate
        # ------------------------------------------------------------------- #
        ax.append( subplot(3,1,3, sharex=ax[0]) );
        grid(color='0.95', linestyle='-')
        ax[-1].set_xscale('log', nonposx='clip')
        #
        plot( this.f[pos_mask], this.fd_dphi[pos_mask], color=sqrt(clr[0]) )
        pylim( this.f[pos_mask], this.fd_dphi[pos_mask] )

        # ------------------------------------------------------------------- #
        # Full figure settings
        # ------------------------------------------------------------------- #
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
        xlabel(r'$f$',fontsize=fs,color=txclr)

        #
        if show:
            shw()

        #
        return ax

    # Plot time domain
    def plottd( this,
              show=False,
              fig = None,
              title = None):

        #
        import warnings
        from numpy import array

        #
        from matplotlib.pyplot import plot,subplot,figure,tick_params,subplots_adjust
        from matplotlib.pyplot import grid,setp,tight_layout,margins,xlabel
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import ylabel as yl
        from matplotlib.pyplot import title as ttl
        from numpy import ones,sqrt,hstack

        #
        if fig is None:
            fig = figure(figsize = 1.1*array([8,7.2]))
            fig.set_facecolor("white")

        #
        clr = rgb(3)
        grey = 0.9*ones(3)
        lwid = 1
        txclr = 'k'
        fs = 18
        font_family = 'serif'

        #
        ax = []
        xlim = lim(this.t) # [-400,this.t[-1]]

        # Time domain plus and cross parts
        ax.append( subplot(3,1,1) );
        grid(color='0.95', linestyle='-')
        setp(ax[-1].get_xticklabels(), visible=False)
        # actual plotting
        plot( this.t, this.plus,  linewidth=lwid, color=0.8*grey )
        plot( this.t, this.cross, linewidth=lwid, color=0.5*grey )
        plot( this.t, this.amp,   linewidth=lwid, color=clr[0] )
        plot( this.t,-this.amp,   linewidth=lwid, color=clr[0] )

        # Ignore renderer warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight_layout(pad=2, w_pad=1.2)
            subplots_adjust(hspace = .001)

        #
        pylim( this.t, this.amp, domain=xlim, symmetric=True )
        kind = this.kind
        yl(kind,fontsize=fs,color=txclr, family=font_family )

        # Time domain phase
        ax.append( subplot(3,1,2, sharex=ax[0]) );
        grid(color='0.95', linestyle='-')
        setp(ax[-1].get_xticklabels(), visible=False)
        # actual plotting
        plot( this.t, this.phi, linewidth=lwid, color=1-clr[0] )
        pylim( this.t, this.phi, domain=xlim )
        yl( r'$\phi = \mathrm{arg}(%s)$' % kind.replace('$','') ,fontsize=fs,color=txclr, family=font_family)

        # Time domain frequency
        ax.append( subplot(3,1,3, sharex=ax[0]) );
        grid(color='0.95', linestyle='-')
        # Actual plotting
        plot( this.t, this.dphi, linewidth=lwid, color=sqrt(clr[0]) )
        pylim( this.t, this.dphi, domain=xlim )
        yl(r'$\mathrm{d}{\phi}/\mathrm{d}t$',fontsize=fs,color=txclr, family=font_family)

        # Full figure settings
        ax[0].set_xlim(lim(this.t))
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

        # Set this object's window
        this.window = this.window * window

        #
        wfarr = this.wfarr
        wfarr[:,1] = this.window * this.wfarr[:,1]
        wfarr[:,2] = this.window * this.wfarr[:,2]

        # NOTE that objects cannot be redefined within their methods, but their properties can be changed. For this reason, the line below uses setfields() rather than gwf() to apply the taper.
        this = this.setfields( wfarr=wfarr )

    # Apply mask
    def apply_mask( this, mask=None ):
        #
        if mask is None: error('the mask input must be given, and it must be index or boolean ')
        #
        this.setfields( this.wfarr[mask,:] )

    # If desired, reset the waveform object to its original state (e.g. it's state just afer loading).
    # Note that after this methed is called, the current object will occupy a different address in memory.
    def reset(this): this.setfields( this.__rawgwfarr__ )

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
    def pad(this,new_length=None):

        # Pad this waveform object to the left and right with zeros
        if new_length is not None:
            # Create the new wfarr
            wfarr = pad_wfarr( this.wfarr, new_length )
            # Confer to the current object
            this.setfields(wfarr)


# Class for waveforms: Psi4 multipoles, strain multipoles (both spin weight -2), recomposed waveforms containing h+ and hx. NOTE that detector response waveforms will be left to pycbc to handle
class gwylm:

    # Class constructor
    def __init__( this,                             # reference for the object to be created
                  scentry_obj,                      # member of the scentry class
                  lm                    = None,     # iterable of length 2 containing multipolr l and m
                  lmax                  = None,     # if set, multipoles with all |m| up to lmax will be loaded.
                                                    # This input is not compatible with the lm tag
                  dt                    = None,     # if given, the waveform array will beinterpolated to
                                                    # this timestep
                  load                  = True,     # IF true, we will try to load data from the scentry_object
                  clean                 = False,    # Toggle automatic tapering
                  extraction_parameter  = None,     # Extraction parameter labeling extraction zone/radius for run
                  level = None,                     # Opional refinement level for simulation. NOTE that not all NR groups use this specifier. In such cases, this input has no effect on loading.
                  w22 = None,                       # Optional input for lowest physical frequency in waveform; by default an wstart value is calculated from the waveform itself and used in place of w22
                  verbose               = None ):   # be verbose

        # Print non None inputs to screen
        thisfun = this.__class__.__name__
        this.verbose = verbose
        if verbose is not None:
            for k in dir():
                if (eval(k) is not None) and not ('this' in k):
                    msg = 'Found %s (=%r) keyword.' % (textul(k),eval(k))
                    alert( msg, thisfun )

        # validate the lm input
        this.__valinputs__(thisfun,lm=lm,lmax=lmax,scentry_obj=scentry_obj)

        # Confer the scentry_object's attributes to this object for ease of referencing
        for attr in scentry_obj.__dict__.keys():
            setattr( this, attr, scentry_obj.__dict__[attr] )

        ''' Explicitely reconfigure the scentry object for the current user. '''
        # this.config.reconfig() # NOTE that this line is commented out because scentry_obj.simdir() below calls the reconfigure function internally.

        # Tag this object with the simulation location of the given scentry_obj. NOTE that the right hand side of this assignment depends on the user's configuration file. Also NOTE that the configuration object is reconfigured to the system's settings within simdir()
        this.simdir = scentry_obj.simdir()

        # Load default values for extraction_parameter and level (e.g. resolution level)
        default_extraction_par = this.config.default_par_list[0]
        default_level = this.config.default_par_list[1]
        # If no extraction parameter is given, retrieve default. NOTE that this depends on the current user's configuration.
        if extraction_parameter is None:
            extraction_parameter = default_extraction_par
        if level is None:
            level = default_level
        # Store the extraction parameter and level
        this.extraction_parameter = extraction_parameter
        this.level = level

        # These fields are initiated here for visiility, but they are filled as lists of gwf object in load()
        this.ylm,this.hlm,this.flm = [],[],[] # psi4 (loaded), strain(calculated by default), news(optional non-default)

        # time step
        this.dt = dt

        # Load the waveform data
        if load==True: this.__load__(lmax=lmax,lm=lm)

        # Characterize the waveform's start and store related information to this.starting
        this.starting = None # In charasterize_start(), the information about the start of the waveform is actually stored to "starting". Here this field is inintialized for visibility.
        this.characterize_start()

        # If w22 is input, then use the input value for strain calculation. Otherwise, use the algorithmic estimate.
        if w22 is None:
            w22 = this.wstart_pn
            if verbose:
                # msg = 'Using w22 from '+bold(magenta('algorithmic estimate'))+' to calculate strain multipoles.'
                msg = 'Using w22 from a '+bold(magenta('PN estimate'))+' to calculate strain multipoles [see pnw0 in basics.py, and/or arxiv:1310.1528v4].'
                alert(msg,thisfun)
        else:
            if verbose:
                msg = 'Using w22 from '+bold(magenta('user input'))+' to calculate strain multipoles.'
                alert(msg,thisfun)

        # Calculate strain
        this.calchlm(w22=w22)

        # Clean the waveforms of junk radiation if desired
        this.__isclean__ = False
        if clean:
            this.clean()

        # Create a dictionary representation of the mutlipoles
        this.lm = {}
        for k,y in enumerate(this.ylm):
            this.lm[(y.l,y.m)] = { 'psi4':y, 'strain':this.hlm[k] }

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
        if not isinstance(scentry_obj,scentry):
            msg = 'First input must be member of scentry class (e.g. as returned from scsearch() ).'
            error(msg,thisfun)

    # Wrapper for core load function. NOTE that the extraction parameter input is independent of the usage in the class constructor.
    def __load__( this,                      # The current object
                  lmax=None,                 # max l to use
                  lm=None,                   # (l,m) pair or list of pairs to use
                  extraction_parameter=None, # the label for different extraction zones/radii
                  level = None,              # Simulation resolution level (Optional and not supported for all groups )
                  dt=None,
                  verbose=None ):

        #
        from numpy import shape

        # If if an lmax value is given.
        if lmax is not None:
            # Then load all multipoles within lmax
            for l in range(2,lmax+1):
                #
                for m in range(-l,l+1):
                    #
                    this.load(lm=[l,m],dt=dt,extraction_parameter=extraction_parameter,level=level,verbose=verbose)
        else: # Else, load the given lis of lm values
            # If lm is a list of specific multipole indeces
            if len(shape(lm))==2:
                #
                for k in lm:
                    if len(k)==2:
                        this.load(lm=k,extraction_parameter=extraction_parameter,level=level,dt=dt)
                    else:
                        msg = 'Found list of multipole indeces (e.g. [[2,2],[3,3]]), but length of one of the index values is not two. Please check your lm input.'
                        error(msg,'gwylm.__load__')
            else: # Else, if lm is a single mode index
                #
                this.load(lm=lm,extraction_parameter=extraction_parameter,level=level,dt=dt)

    # load the waveform data
    def load(this,                  # The current object
             lm=None,               # the l amd m values of the multipole to load
             file_location=None,    # (Optional) is give, this file string will be used to load the file,
                                    # otherwise the function determines teh file string automatically.
             dt = None,             # Time step to enforce for data
             extraction_parameter=None,
             level=None,            # (Optional) Level specifyer for simulation. Not all simulation groups use this!
             output=False,          # Toggle whether to store data to the current object, or output it
             verbose=None):

        # Import useful things
        from os.path import isfile,basename
        from numpy import sign,diff,unwrap,angle,max
        from scipy.stats.mstats import mode
        from scipy.version import version as scipy_version
        thisfun=inspect.stack()[0][3]

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
            if this.verbose: alert('Loading: %s' % cyan(basename(file_location)), thisfun )
            wfarr,_ = smart_load( file_location, verbose=this.verbose )

            # Initiate waveform object and check that sign convetion is in accordance with core settings
            def mkgwf(wfarr_):
                return gwf( wfarr_,
                            l=l,
                            m=m,
                            extraction_parameter=extraction_parameter,
                            dt=dt,
                            kind='$rM\psi_{%i%i}$'%(l,m))

            #
            y_ = mkgwf(wfarr)

            # ---------------------------------------------------- #
            # Enforce internal sign convention for Psi4 multipoles
            # ---------------------------------------------------- #

            msk_ = y_.amp > 0.01*max(y_.amp)
            if int(scipy_version.split('.')[1])<16:
                # Account for old scipy functionality
                external_sign_convention = sign(m) * mode( sign( y_.dphi[msk_] ) )[0][0]
            else:
                # Account for modern scipy functionality
                external_sign_convention = sign(m) * mode( sign( y_.dphi[msk_] ) ).mode[0]

            if M_RELATIVE_SIGN_CONVENTION != external_sign_convention:
                wfarr[:,2] = -wfarr[:,2]
                y_ = mkgwf(wfarr)
                # Let the people know what is happening.
                msg = yellow('Re-orienting waveform phase')+' to be consistent with internal sign convention for Psi4, where sign(dPhi/dt)=%i*sign(m).' % M_RELATIVE_SIGN_CONVENTION + ' Note that the internal sign convention is defined in ... nrutils/core/__init__.py as "M_RELATIVE_SIGN_CONVENTION". This message has appeared becuase the waveform is determioned to obey and sign convention: sign(dPhi/dt)=%i*sign(m).'%(external_sign_convention)
                thisfun=inspect.stack()[0][3]
                alert( msg, thisfun )

            # use array data to construct gwf object with multipolar fields
            if not output:
                this.ylm.append( y_ )
            else:
                return y_

        else:

            # There has been an error. Let the people know.
            msg = '(!!) Cannot find "%s". Please check that catalog_dir and data_file_name_format in %s are as desired. Also be sure that input l and m are within ranges that are actually present on disk.' % ( red(file_location), magenta(this.config.config_file_location) )
            raise NameError(msg)

    # Plotting function for class: plot plus cross amp phi of waveforms USING the plot function of gwf()
    def plot(this,show=False,fig=None,kind=None,verbose=False,domain=None):
        #
        from matplotlib.pyplot import show as shw
        from matplotlib.pyplot import figure
        from numpy import array,diff,pi

        #
        if fig is None and kind!='both':
            fig = figure( figsize = 1.1*array([8,7.2]) )
            fig.set_facecolor("white")

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
                    msg = '(**) You have requested that strain be plotted before having explicitelly called MMRDNSlm.calchlm(). I will now call calchlm() for you.'
                    print magenta(msg)
                    this.calchlm()
                # Assign strain to the general placeholder.
                wflm = this.hlm

            # Plot waveform data
            for y in wflm:
                ax = y.plot(fig=fig,title='%s: %s' % (this.setname,this.label),domain=domain)

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
                ax = this.plot(show=show,kind=kind,domain=domain)

        #
        return ax

    # Strain via ffi method
    def calchlm(this,w22=None):

        # Calculate strain according to the fixed frequency method of http://arxiv.org/pdf/1006.1632v3

        #
        from numpy import array,double

        # If there is no w22 given, then use the internally defined value of wstart
        if w22 is None:
            # w22 = this.wstart
            # NOTE: here we choose to use the ORBITAL FREQUENCY as a lower bound for the l=m=2 mode.
            w22 = this.wstart_pn


        # Reset
        this.hlm = []
        for y in this.ylm:

            # Calculate the strain for each part of psi4. NOTE that there is currently NO special sign convention imposed beyond that used for psi4.
            w0 = w22 * double(y.m)/2.0 # NOTE that wstart is defined in characterize_start() using the l=m=2 Psi4 multipole.
            # Here, m=0 is a special case
            if 0==y.m: w0 = w22
            # Let the people know
            if this.verbose:
                print magenta('* w0(w22) = %f' % w0)+yellow(' (this is the lower frequency used for FFI method [arxiv:1006.1632v3])')

            # Create the core waveform information
            t       =  y.t
            h_plus  =  ffintegrate( y.t, y.plus,  w0, 2 )
            h_cross =  ffintegrate( y.t, y.cross, w0, 2 )

            # Constrcut the waveform array for the new strain object
            wfarr = array( [ t, h_plus, h_cross ] ).T

            # Add the new strain multipole to this object's list of multipoles
            this.hlm.append( gwf( wfarr, l=y.l, m=y.m, kind='$rh_{%i%i}/M$'%(y.l,y.m) ) )

    # Characterise the start of the waveform using the l=m=2 psi4 multipole
    def characterize_start(this):

        # Look for the l=m=2 psi4 multipole
        y22_list = filter( lambda y: y.l==y.m==2, this.ylm )
        # If it doesnt exist in this.ylm, then load it
        if 0==len(y22_list):
            y22 = this.load(lm=[2,2],output=True)
        else:
            y22 = y22_list[0]
        # Use the l=m=2 psi4 multipole to determine the waveform start
        # store information about the start of the waveform to the current object
        this.starting = gwfcharstart( y22 )
        # store the expected min frequency in the waveform to this object as:
        this.wstart = this.starting.left_dphi
        this.startindex = this.starting.left_index
        # Estimate the smallest orbital frequency relevant for this waveform using a PN formula.
        safety_factor = 0.90
        this.wstart_pn = safety_factor*2.0*pnw0(this.m1,this.m2,this.b)

    # Clean the time domain waveform by removing junk radiation.
    def clean( this, method=None, crop_time=None ):

        # Default cleaning method will be smooth windowing
        if method is None:
            method = 'window'

        # ---------------------------------------------------------------------- #
        # A. Clean the start of the waveform using information from the characterize_start method
        # ---------------------------------------------------------------------- #

        if not this.__isclean__ :

            if method.lower() == 'window':

                # Calculate the window to be applied using the starting information. The window nwill be aplied equally to all multipole moments. NOTE: language disambiguation -- a taper is the part of a window that varies from zero to 1 (or 1 to zero); a window may contain many tapers. Also NOTE that the usage of this4[0].ylm[0].t below is an arbitration -- any array of the dame dimentions could be used.
                window = maketaper( this.ylm[0].t, [this.starting.left_index,this.starting.right_index] )

                # Apply this window to both the psi4 and strain multipole moments. The function, taper(), is a method of the gwf class.
                for y in this.ylm:
                    y.apply_window( window=window )
                for h in this.hlm:
                    h.apply_window( window=window )

            elif method.lower() == 'crop':

                # Crop such that the waveform daya starts abruptly
                from numpy import arange,double

                if not (crop_time is None):
                    # If there is no crop time given, then use the low frequency value given by the nrutils start characterization time
                    mask = arange( this.startindex, this.ylm[0].n )
                elif isinstance(crop_time,(double,int,float)):
                    # Otherwise, use an input starting time
                    mask = this.ylm[0].raw[:,0] > crop_time

                for y in this.ylm:
                    y.apply_mask( mask )
                for h in this.hlm:
                    h.apply_mask( mask )

            # ---------------------------------------------------------------------- #
            # B. Clean the end of the waveform using information from the characterize_end method.
            # ---------------------------------------------------------------------- #

            # # TODO: implement this.
            # window = makewindow( this.t, [this.ending.left_index,this.ending.right_index] )
            # for y in this.ylm:
            #     y.taper( window=this.window )
            # for h in this.hlm:
            #     h.taper( window=this.window )

            #
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
                w0 = w22 * double(y.m)/2.0 # NOTE that wstart is defined in characterize_start() using the l=m=2 Psi4 multipole.
                # Here, m=0 is a special case
                if 0==y.m: w0 = w22
                # Let the people know
                if this.verbose:
                    print magenta('* w0(w22) = %f' % w0)+yellow(' (this is the lower frequency used for FFI method [arxiv:1006.1632v3])')

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
                warning(msg,'gwylm.calcflm')

            # Store the flm list to the current object
            this.flm = flm


    #--------------------------------------------------------------------------------#
    # Get a gwylm object that only contains ringdown
    #--------------------------------------------------------------------------------#
    def ringdown(this,              # The current object
                 T0 = 10,           # Starting time relative to peak luminosity of the l=m=2 multipole
                 df = None,         # Optional df in frequency domain (determines time domain padding)
                 verbose = True):

        #
        from numpy import linspace,array
        from scipy.interpolate import InterpolatedUnivariateSpline as spline

        # Make sure that the l=m=2 multipole exists
        if not ( (2,2) in this.lm.keys() ):
            msg = 'There must be a l=m=2 multipole prewsent to estimate the waveform\'s ringdown part.'
            error(msg,'gwylm.ringdown')

        # Use the l=m=2 multipole to estimate the luminosity.
        this.calcflm()

        # Retrieve the l=m=2 component
        f = [ a for a in this.flm if a.l==a.m==2 ][0]

        # Use its time series to define a mask
        a = f.intrp_t_amp_max+T0
        b = f.t[-1]
        n = abs(float(b-a))/f.dt
        t = linspace(a,b,n)

        #
        that = this.copy()

        #
        def __ringdown__(wlm):
            #
            xlm = []
            for k,y in enumerate(wlm):
                # Create interpolated plus and cross parts
                plus  = spline(y.t,y.plus)(t)
                cross = spline(y.t,y.cross)(t)
                # Create waveform array
                wfarr = array( [t-f.intrp_t_amp_max,plus,cross] ).T
                # Create gwf object
                xlm.append(  gwf(wfarr,l=y.l,m=y.m,kind=y.kind)  )
            #
            return xlm
        #
        that.ylm = __ringdown__( this.ylm )
        that.flm = __ringdown__( this.flm )
        that.hlm = __ringdown__( this.hlm )
        that.characterize_start()

        #
        return that


    # pad each mode to a new_length
    def pad(this,new_length=None):

        # Pad each mode
        for y in this.ylm:
            y.pad( new_length=new_length )
        for h in this.hlm:
            h.pad( new_length=new_length )


    # Extrapolate to infinite radius: http://arxiv.org/pdf/1503.00718.pdf
    def extrapolate(this,method=None):

        msg = 'This method is under development and cannot currently be used.'
        error(msg,'gwylm.extrapolate')

        # If the simulation is already extrapolated, then do nothing
        if this.__isextrapolated__:
            # Do nothing
            print
        else: # Else, extrapolate
            # Use radius only scaling
            print

        return None



# Time Domain LALSimulation Waveform Approximant h_pluss and cross, but using nrutils data conventions
def lswfa( apx      ='IMRPhenomPv2',    # Approximant name; must be compatible with lal convenions
           q        = None,           # mass ratio > 1
           S1       = None,           # spin1 iterable
           S2       = None,           # spin2 iterable
           fmin_hz  = 30.0,           # phys starting freq in Hz
           verbose  = False ):        # boolean toggle for verbosity

    #
    from numpy import array,linspace,double
    import lalsimulation as lalsim
    import lal

    # Standardize input mass ratio and convert to component masses
    M = 70.0
    q = double(q)
    q = max( [q,1.0/q] )
    m2 = M * 1.0 / (1.0+q)
    m1 = float(q) * m2

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


# Characterize the START of a time domain waveform
class gwfcharstart:

    #
    def __init__( this,                 # the object to be created
                  y,                    # input gwf object who'se start behavior will be characterised
                  shift     = 2,        # The size of the turn on region in units of waveform cycles.
                  verbose   = False ):

        #
        from numpy import arange,diff,where,array,ceil,mean
        from numpy import histogram as hist
        thisfun=this.__class__.__name__

        # Take notes on what happens
        notes = []

        # This algorithm estimates the start of the gravitational waveform -- after the initial junk radiation that is present within most raw NR output. The algorithm proceeds in the manner consistent with a time domain waveform.

        # Validate inputs
        if not isinstance(y,gwf):
            msg = 'First imput must be a '+cyan('gwf')+' object. Type %s found instead.' % type(y).__name__
            error(msg,thisfun)

        # If the waveform starts at its peak (e.g. in the case of ringdown)
        if not y.k_amp_max:

            #
            this.left_index = 0
            this.right_index = 0
            this.left_dphi=this.center_dphi=this.right_dphi = y.dphi[this.right_index]
            this.peak_mask = [0]

        else:

            # 1. Find the pre-peak portion of the waveform.
            val_mask = arange( y.k_amp_max )
            # 2. Find the peak locations of the plus part.
            pks,pk_mask = findpeaks( y.cross[ val_mask ] )
            pk_mask = pk_mask[ pks > y.amp[y.k_amp_max]*5e-4 ]

            # 3. Find the difference between the peaks
            D = diff(pk_mask)
            # 4. Find location of the first peak that is separated from its adjacent by greater than the largest value. This location is stored to start_map.
            start_map = find(  D >= max(D)  )[0]

            # 5. Determine the with of waveform turn on in indeces based on the results above. NOTE that the width is bound below by half the difference betwen the wf start and the wf peak locations.
            index_width = min( [ 1+pk_mask[start_map+shift]-pk_mask[start_map], 0.5*(1+y.k_amp_max-pk_mask[ start_map ]) ] )
            # 6. Estimate where the waveform begins to turn on. This is approximately where the junk radiation ends. Note that this area will be very depressed upon windowing, so is can be
            j_id = pk_mask[ start_map ]

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


# Characterize the END of a time domain waveform: Where is the noise floor?
def gwfend():
    #
    return None


# Function which converts lalsim waveform to gwf object
def lalsim2gwf( hp,hc,M,D ):

    #
    from numpy import linspace,array,double,sqrt,hstack,zeros
    from nrutils.tools.unit.conversion import codeh

    # Extract plus and cross data. Divide out contribution from spherical harmonic towards NR scaling
    x = sYlm(-2,2,2,0,0)
    h_plus  = hp.data.data/x
    h_cross = hc.data.data/x

    # Create time series data
    t = linspace( 0.0, (h_plus.size-1.0)*hp.deltaT, int(h_plus.size) )

    # Create waveform
    harr = array( [t,h_plus,h_cross] ).T

    # Convert to code units, where Mtotal=1
    harr = codeh( harr,M,D )

    # Create gwf object
    h = gwf( harr, kind=r'$h^{\mathrm{lal}}_{22}$' )

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


# Shift a gwf object in time by some abount t0
def gwftshift( y,               # the gwf object to be shifted
               t0,              # Amount to shift the object
               verbose=None):   # Whether or not to let the people know

    #
    from numpy import rem

    #
    if type(y).__name__!='gwf':
        msg = 'input must be gwf object'
        error(msg,'gwftshift')

    #
    T = y.t(-1) - y.t(0)
    t0 = rem( t0, T )

    #
    a = y.arr
