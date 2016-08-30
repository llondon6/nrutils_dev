#!/usr/bin/env python
'''
 sxs_download.py

 This script is to download waveforms from the open SXS collaboration website.

 ~ llondon2'14

'''


# Import Libs
import os,shutil,glob,urllib2,tarfile,sys,errno
import time,subprocess,re,inspect,pickle
import numpy,string,random,h5py,copy


# Class for basic print manipulation
class print_format:
   purple = '\033[95m'
   cyan = '\033[96m'
   darkcyan = '\033[36m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bold = '\033[1m'
   ul = '\033[4m'
   end = '\033[0m'

# Function that uses the print_format class to make tag text for bold printing
def bold(str):
    return print_format.bold + str + print_format.end
def red(str):
    return print_format.red + str + print_format.end
def green(str):
    return print_format.green + str + print_format.end
def blue(str):
    return print_format.blue + str + print_format.end

#
def parent(path):
    '''
    Simple wrapper for getting parent directory
    '''
    return os.path.abspath(os.path.join(path, os.pardir))+'/'


# Make "mkdir" function for directories
def mkdir(dir):
    # Check for directory existence; make if needed.
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Return status
    return os.path.exists(dir)


# Function that returns true if for string contains l assignment that is less than l_max
def l_test(string,l_max):
    '''
    Function that returns true if for string contains l assignment that is less than l_max:
    score = ltest('Ylm_l3_m4_stuff.asc',3)
          = True
    score = ltest('Ylm_l3_m4_stuff.asc',5)
          = True
    score = ltest('Ylm_l3_m4_stuff.asc',2)
          = True
    '''
    # break string into bits by l
    score = False
    for bit in string.split('l'):
        if bit[0].isdigit():
            score = score or int( bit[0] )<= l_max

    # return output
    return score

#
def h5tofiles( h5_path, save_dir, file_filter= lambda s: True, cleanup = False, prefix = '' ):
    '''
    Function that takes in h5 file location, and and writes acceptable contents to files using groups as directories.
    ~ lll2'14
    '''

    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    #
    def group_to_files( group, work_dir ):
        '''
        Recurssive fucntion to make folder trees from h5 groups and files.
        ~ lll2'14
        '''

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        if type(group) is h5py._hl.group.Group or \
           type(group) is h5py._hl.files.File:
            # make a directory with the group name
            this_dir = work_dir + group.name.split('.')[0]
            if this_dir[-1] is not '/': this_dir = this_dir + '/'
            mkdir( this_dir )
            #
            for key in group.keys():
                #
                if type(group[key]) is h5py._hl.group.Group or \
                   type(group[key]) is h5py._hl.files.File:
                    #
                    group_to_files( group[key], this_dir )
                elif type(group[key]) is h5py._hl.dataset.Dataset:
                    #
                    data_file_name = prefix + key.split('.')[0]+'.asc'
                    if file_filter( data_file_name ):
                        #
                        data_file_path = this_dir + data_file_name
                        #
                        data = numpy.zeros( group[key].shape )
                        group[key].read_direct(data)
                        #
                        print( '[%s]>> ' % thisfun + bold('Writing') + ': "%s"'% data_file_path)
                        numpy.savetxt( data_file_path, data, delimiter="  ", fmt="%20.8e")
                else:
                    #
                    raise NameError('Unhandled object type: %s' % type(group[key]))
        else:
            #
            raise NameError('Input must be of the class "h5py._hl.group.Group".')

    #
    if os.path.isfile( h5_path ):

        # Open the file
        h5_file = h5py.File(h5_path,'r')

        # Begin pasing each key, and use group to recursively make folder trees
        for key in h5_file.keys():

            # reset output directory
            this_dir = save_dir

            # extract reference object with h5 file
            ref = h5_file[ key ]

            # If the key is a group
            if type(ref) is h5py._hl.group.Group:

                #
                group_to_files( ref, this_dir )


            else: # Else, if it's a writable object

                print('[%s]>> type(%s) = %s' % (thisfun,key,type(ref)) )

        # If the cleanup option is true, delete the original h5 file
        if cleanup:
            #
            print('[%s]>> Removing the original h5 file at: "%s"' % (thisfun,h5_path) )
            os.remove(h5_path)

    else:

        # Raise Error
        raise NameError('No file at "%s".' % h5_path)


# file_path = '/home/janus/WORK/Data/Waveforms/SXS/SXS0001/rPsi4_FiniteRadii_CodeUnits.h5'
#
# h5tofiles( file_path, parent(file_path), file_filter= lambda s: l_test(s,5)  )
#
# #
# tree = h5py.File(file_path,'r')
#
# j=0
# for item in tree.keys():
#     print '\t'*0 + item + ":", tree[item]
#     k=0
#     for branch in tree[item]:
#         k+=1
#         print '\t'*1 + branch + ": ",tree[item][branch]
#     j+=1
#     if j==1:
#         break

#
def replace_line(file_path, pattern, substitute, **kwargs):
    '''
    Function started from: https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python.

    This function replaces an ENTIRE line, rather than a string in-line.

    ~ ll2'14
    '''

    #
    from tempfile import mkstemp
    from shutil import move
    # Get the string for this function name
    thisfun = inspect.stack()[0][3]

    # Look for verbose key
    keys = ('verbose','verb')
    VERB = parsin( keys, kwargs )
    if VERB:
        print('[%s]>> VERBOSE mode on.' % thisfun)

    #
    if substitute[-1] is not '\n':
        substitute = substitute + '\n'

    # If the file exists
    if os.path.isfile(file_path):
        #
        if VERB:
            print( '[%s]>> Found "%s"' % (thisfun,file_path) )
        # Create temp file
        fh, abs_path = mkstemp()
        if VERB: print( '[%s]>> Temporary file created at "%s"' % (thisfun,abs_path) )
        new_file = open(abs_path,'w')
        old_file = open(file_path)
        for line in old_file:
            pattern_found = line.find(pattern) != -1
            if pattern_found:
                if VERB:
                    print( '[%s]>> Found pattern "%s" in line:\n\t"%s"' % (thisfun,pattern,line) )
                new_file.write(substitute)
                if VERB:
                    print( '[%s]>> Line replaced with:\n\t"%s"' % (thisfun,substitute) )
            else:
                new_file.write(line)
        # Close temp file
        new_file.close()
        os.close(fh)
        old_file.close()
        # Remove original file
        os.remove(file_path)
        # Move new file
        move(abs_path, file_path)
        # NOTE that the temporary file is automatically removed
        if VERB: print( '[%s]>> Replacing original file with the temporary file.' % (thisfun) )
    else:
        #
        if VERB:
            print( '[%s]>> File not found at "%s"' % (thisfun,file_path) )
        if VERB:
            print( '[%s]>> Creating new file at "%s"' % (thisfun,file_path) )
        #
        file = open( file_path, 'w' )
        if substitute[-1]!='\n':
            substitute = substitute + '\n'
        #
        if VERB:
            print( '[%s]>> Writing "%s"' % (thisfun,substitute) )
        #
        file.write(substitute)
        file.close()
    #
    if VERB:
        print('[%s] All done!',thisfun)

# Function that returns randome strings of desired length and component of the desired set of characters
def rand_str(size=2**4, characters=string.ascii_uppercase + string.digits):
    '''
    Function that returns randome strings of desired length and component of the desired set of characters. Started from: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    -- ll2'14
    '''
    # Ensure that each character has the same probability of being selected by making the set unique
    characters = ''.join(set(characters))
    # return the random string
    return ''.join(random.choice(characters) for _ in range(size))

#
def parsin( keys, dict, default=False, verbose=False, fname='*', **kwarg ):
    '''
    Function for interpretive keyword parsing:
    1. Given the dictionary arguments of a fuction,
    scan for a member of the set "keys".
    2. If a set member is found, output it's dictionary reference.
    The net result is that multiple keywords can be mapped to a
    single internal keyword for use in the host function. Just as traditional
    keywords are initialized once, this function should be used within other
    functions to initalize a keyword only once.
    -- ll2'14
    '''

    if type(keys)==str:
        keys = [keys]

    # print('>> Given key list of length %g' % len(keys))
    value = default
    for key in keys:
        #if verbose:
        #    print('>> Looking for "%s" input...' % key)
        if key in dict:

            if verbose:
                print('[%s]>> Found "%s" or variant thereof.' % (fname,key) )

            value = dict[key]
            break
    #
    return value

# Rough grep equivalent using the subprocess module
def grep( flag, file_location, options='', comment='' ):
    # Create string for the system command
    cmd = "grep " + '"' + flag + '" ' + file_location + options
    # Pass the command to the operating system
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    raw_output = process.communicate()[0]
    # Split the raw output into a list whose elements are the file's lines
    output = raw_output.splitlines()
    # Mask the lines that are comments
    if comment:
        # Masking in Python:
        mask = [line[0]!=comment for line in output]
        output = [output[k] for k in xrange(len(output)) if mask[k]]

    # Return the list of lines
    return output

# Simple function to determine whether or not a string is intended to be a
# number: all numbers are composed of a set dictionary of characters
def isnumeric( str ):
    for mark in list('12345678901e- ,][.'):
        str = str.replace(mark,'')
    return len(str)==0

# Rudimentary function for printing text in the center of the terminal window
def center_space(str):
    x = os.popen('stty size', 'r').read()
    if x:
        rows, columns = x.split()
        a = ( float(columns) - float(len(str)+1.0) ) /2.0
    else:
        a = 0
    return ' '*int(a)
def center_print(str):
    pad = center_space(str)
    print pad + str

# Print a short about statement to the prompt
def print_hl(symbol="<>"):
    '''
    Simple function for printing horizontal line across terminal.
    ~ ll2'14
    '''
    x = os.popen('stty size', 'r').read()
    if x:
        rows, columns = x.split()
        if columns:
            print symbol*int(float(columns)/float(len(symbol)))

# Function for untaring datafiles
def untar(tar_file,savedir='',verbose=False,cleanup=False):
    # Save to location of tar file if no other directory given
    if not savedir:
        savedir = os.path.dirname(tar_file)
    # Open tar file and extract
    tar = tarfile.open(tar_file)
    internal_files = tar.getnames()
    tar.extractall(savedir)
    tar.close()
    if verbose:
        print ">> untar: Found %i files in tarball." % len(internal_files)
    if cleanup:
        os.remove(tar_file)

# Function for file downloading from urls
def download( url, save_path='', save_name='', size_floor=[], verbose=False, overwrite=True ):

    # set default file name for saving
    if not save_name:
        save_name = url.split('/')[-1]

    # Create full path of file that will be downloaded using URL
    path,file_type = os.path.splitext(url)
    file_location = save_path + save_name
    u = urllib2.urlopen(url)

    # Determine whether the download is desired
    DOWNLOAD = os.path.isfile(file_location) and overwrite
    DOWNLOAD = DOWNLOAD or not os.path.isfile(file_location)

    # Set the default output
    done = False

    #
    if DOWNLOAD:
        f = open(file_location, 'wb')
        file_size_dl = 0
        block_sz = 10**4 # bites
        # Time the download by getting the current system time
        t0 = time.time()
        # Perform the download
        k=0
        while True:
            t1 = time.time();
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            mb_downloaded = file_size_dl/(10.0**6.0);
            dt = time.time() - t1
            if k==0:
                status = r"   Download Progress:%1.2f MB downloaded" % mb_downloaded
            else:
                status = r"   Download Progress:%1.2f MB downloaded at %1.2f Mb/sec     " % (mb_downloaded,(len(buffer)/(10.0**6.0))/dt)
            status = status + chr(8)*(len(status)+1)
            k += 1
            if verbose: print status,
        # Close file
        f.close()
        # Get the final time
        tf = time.time()
        # Show completion notice
        if verbose: print "   Download of %1.4f MB completed in %1.4f sec" % ((file_size_dl/(10.0**6.0)),tf-t0)
        if verbose: print "   Average download rate: %1.4f Mb/sec" % ((file_size_dl/(10.0**6.0))/(tf-t0))
        if verbose: print('   Saving:"%s"' % file_location )
        # If the size of this file is below the floor, delete it.
        if size_floor:
            if file_size_dl<size_floor:
                os.remove(file_location)
                if verbose: print( '   *File is smaller than %i bytes and has been deleted.' % size_floor )
                done = True
    else:
        #
        print('   *File exists and overwrite is not turned on, so this file will be skipped.')

    return (done,file_location)




# Class for dynamic data objects such as sim-catalog-entries (scentry's)
class smart_object:
    '''
    This class has the ability to learn files and string by making file elemnts
    its attributes and automatically setting the attribute values.
    ~ll2'14
    '''

    def __init__(this,attrfile=None,id=None,**kwargs):
        #
        this.valid = False
        this.source_file_path = []
        this.source_dir  = []

        #
        if attrfile is not None:
            this.learn_file( attrfile, **kwargs )

    #
    def show( this ):

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        for attr in this.__dict__.keys():
            value = this.__dict__[attr]
            print( '[%s]>> %s = %s' % (thisfun,attr,str(value)) )

    # Function for parsing entire files into class attributes and values
    def learn_file( this, file_location, eqls="=", **kwargs ):
        # Use grep to read each line in the file that contains an equals sign
        line_list = grep(eqls,file_location,comment='#')
        for line in line_list:
            this.learn_string( line,eqls, **kwargs )
        # Learn file location
        this.source_file_path.append(file_location)
        # Learn location of parent folder
        this.source_dir.append( parent(file_location) )

    # Function for parsing single lines strings into class attributes and values
    def learn_string(this,string,eqls="=",**kwargs):

        #
        from numpy import array

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        # Look for verbose key
        keys = ('verbose','verb')
        VERB = parsin( keys, kwargs )
        if VERB:
            print('[%s]>> VERBOSE mode on.' % thisfun)

        # The string must be of the format "A eqls B", in which case the result is
        # that the field A is added to this object with the value B
        part = string.split(eqls)

        # Remove harmful and unneeded characters
        attr = part[0].replace('-','_')
        attr = attr.replace(' ','')
        attr = attr.replace('#','')
        part[1] = part[1].replace(' ','')
        if VERB: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))

        # Correctly formatted lines will be parsed into exactly two parts
        if [2 == len(part)]:
            #
            value = []
            if part[1].split(','):
                is_number = True
                for val in part[1].split(','):
                    #
                    if  not isnumeric( val ):   # IF
                        is_number = False
                        if VERB: print( '>> Learning character: %s' % val )
                        value.append( val )
                    else:                       # Else
                        if VERB: print( '>> Learning number: %s' % val)
                        if val:
                            value.append( eval(val) )
                #
                if is_number:
                    value = array(value)
            else:
                value.append("none")
            #
            if 1==len(value):
                value = value[0]
            #
            setattr( this, attr, value )
        else:
            raise ValueError('Impoperly formatted input string.')













# Create class for meta data information
class scentry(smart_object):
    '''
    Dynamic class for the representation of simulation meta data or parameter
    files: Members of this class have attributes that correspond to assignments
    in string passed through the "learn_string" method below.
    '''
    def __init__(this,id=[]):
        #
        this.id = id
        this.valid = False
        this.source_file_path = []
        this.source_dir  = []
#
def sc_search(catalog_location,**kwargs):
    ''' Function for searching through simulation catalogs
    using their the dynamically createdattributes of the
    scentry class.
    '''

    # Get the string for this function name
    thisfun = inspect.stack()[0][3]

    #
    thisfile = os.path.splitext(__file__)[0].split('/')[-1]

    # Look for verbose key
    keys = ('verbose','verb')
    VERB = parsin( keys, kwargs )
    if VERB:
        print('[%s]>> '% thisfun+bold('VERBOSE')+' mode on.' )

    # Relative path for settings file
    settings_file_path = os.path.dirname(os.path.realpath(__file__)) + '/'+thisfile+'.ini'
    if VERB:
        print('>> Found Settings file at: '+settings_file_path)

    # Look for Plotting option
    keys = ('plot','plt')
    PLOT = parsin( keys, kwargs )
    if parsin(keys,kwargs) and VERB:
        print('[%s]>> Found "%s" key or variant thereof.' % ( thisfun, bold(keys[0]) ) )

    # Look for key to change settings file
    keys = ['.ini']
    new_defalt_file_path = parsin( keys, kwargs )
    if new_defalt_file_path:
        if VERB:
            print('[%s]>> Found "%s" key or variant thereof.' % ( thisfun, bold(keys[0]) ) )
        pattern = 'DEFAULT_CATALOG_LOCATION='
        substitute = pattern + new_defalt_file_path
        replace_line(settings_file_path, pattern, substitute )

    # Load Settings
    settings = smart_object()
    settings.learn_file(settings_file_path)

    # Load the default catalog
    from os.path import expanduser
    home = expanduser("~")+'/'
    catalog_location = expanduser(catalog_location)
    if os.path.isfile( catalog_location ):
        catalog = pickle.load( open( catalog_location, "rb" ) )
        if VERB: print( '[%s]>> Found default catalog in "%s"' % (thisfun,catalog_location) )
    else:
        raise NameError('Cannot find catalog string in "%s"' % catalog_location)
    # Loading catalog file
    catalog = pickle.load( open( catalog_location, "rb" ) )
    if VERB:
        print('[%s]>> Loading catalog file.' % thisfun)
    # The DEFAULT output of this fucntion is the full catalog
    output = catalog

    # ######################################################################### #
    # Parse non-spinning key; search if key is found
    # ######################################################################### #
    keys = ('non-spinning','nonspinning','nospin','non_spinning')
    if parsin(keys,kwargs) and VERB:
        print('[%s]>> Found "%s" key or variant thereof.' % ( thisfun, bold(keys[0]) ) )
    # If the key is found, apply the relevant filter:
    if parsin(keys,kwargs):
        # Define a function that is true IF the catalog entry is non-spinning
        key_test = lambda y : numpy.linalg.norm(numpy.array(y.initial_spin1)) \
         + numpy.linalg.norm(numpy.array(y.initial_spin1)) < 1e-6
        # Filter the catalog based upon this boolean test
        output = filter( key_test, output )

    # #
    # if VERB:
    #     k = 0; spacer = '    '
    #     table = texttable.Texttable()
    #     table.set_cols_width([8,6,18,6,6,6,8,7,18])
    #     table.header(['Result#','CatID','AltName','m1/m2','|S1|','|S1|',\
    #     '#Orbits','Remnant Mass','Remnant Spin (Kerr Parameter)'])
    #     for Y in output:
    #         k+=1
    #         this_row = [ '%04d'%k , '%04d'%Y.id , Y.alternative_names,\
    #          numpy.round(Y.initial_mass1/Y.initial_mass2,2),\
    #          numpy.round(numpy.linalg.norm(numpy.array(Y.initial_spin1)),4),\
    #          numpy.round(numpy.linalg.norm(numpy.array(Y.initial_spin2)),4),\
    #          Y.number_of_orbits, Y.remnant_mass, numpy.linalg.norm(Y.remnant_spin)/Y.remnant_mass ]
    #         table.add_row(this_row)
    #     #
    #     time.sleep(0.5)
    #     print( '[%s]>> %s:' % (thisfun, bold('Found')) )
    #     print( table.draw() )

    #
    return output

# Function that takes in a directory, downloads meta data there, then saves a catalog file
def get_metadata( sxs_dir, tmpdir, replace_old = True, **kwargs ):
    '''
    # --------------------------------------------------------------------------------- #
    # For all SXS simulations available, download the metadatafiles into a special folder
    # --------------------------------------------------------------------------------- #
    '''
    # Get the name of this functoin
    thisfun = inspect.stack()[0][3]

    # Look for verbose key
    keys = ('verbose','verb')
    VERB = parsin( keys, kwargs )
    if VERB:
        print('[%s]>> VERBOSE mode on.' % thisfun)

    # Alert the user that matadata dl is starting
    if VERB: print '>> '+bold('Starting the download of Meta Data.')

    # Download each metadatafile
    k = 0
    # Define the string format used for each metadata file
    url_format = "http://www.black-holes.org/waveforms/data/DisplayMetadataFile.php/?id=SXS:BBH:%04i"
    # Define holder for meta data class (e.g. scentry objects)
    catalog = []
    # Define smallest allowed file size
    min_bytes = 200
    # Download the files
    while True:

        # Step the counter
        k += 1
        # Use the counter to create a url and other strings
        url = url_format % k
        # Use the inctrement to uniquely name each simulation folder
        sim_name = 'SXS%04i' % k
        # Make a folder for the simulation meta data
        sim_dir = sxs_dir + sim_name + '/'
        mkdir(sim_dir)
        # Make a string for the meta data file name
        file_name = 'metadata.asc'
        # Update the user to what's happening
        if VERB: print('>> Downloading:\t"%s"' % url )

        # Download the metada file to meta_dir;
        # Note that "done" will be true when the last file downloaded is less than the
        # size_floor stipulated below this means that it's just the serve telling us
        # that the desired file could not be found.
        done,file_location = download(url,sim_dir,save_name=file_name,\
                                      size_floor=min_bytes,verbose=True,overwrite=replace_old)
        # if 2 == k:
        #     done=True
        #     print '*** DEBUGGING: Only downloading a few cases for fast turn around!'
        #     break
        
        # Monitor the download stack
        if done: # Break out of the loop
            if VERB: print(bold('>> Downloading of metadata files completed.'))
            # Remove un-needed folder
            os.removedirs(sim_dir)
            break
        else: # Store the location of this meta data file
            # Copy the metadata file to the catalog directory for safe keeping
            shutil.copy(file_location, tmpdir )
            catalog_md_file_location = tmpdir+sim_name+'.md'
            os.rename( tmpdir+file_name , catalog_md_file_location )
            # Create a scentry object with an ID of k
            catalog.append( scentry(k) )
            # Have the scentry object "learn" the meta data file
            catalog[-1].learn_file( file_location )
            # Let the user know what has just happened
            if VERB: print "   *Creating scentry object dynamically from meta data file."

    # --------------------------------------------------------------------------------- #
    # Store catalog to .p file for later usage
    # --------------------------------------------------------------------------------- #
    catalog_location = tmpdir + 'sxs_catalog.p'
    if VERB: print '>> '+bold('Now storing catalog of metadata to:')+' "%s"' % catalog_location
    pickle.dump( catalog, open( catalog_location, "wb" ) )

    # Return the catalog file location
    return catalog_location














# Load the settings file
settings = smart_object()

settings_file_location = os.path.dirname(os.path.realpath(__file__))\
                            +'/'+os.path.basename(__file__).split('.')[0]+'.ini'

#settings_file_location = '/home/janus/KOALA/Scripts/sxsdltool.ini'

settings.learn_file(settings_file_location)


# Clear the prompt
os.system('clear')

#--------------------------------------------------------------------------------- #
# Alert the user about this program
# --------------------------------------------------------------------------------- #
wave_train = '~~~~<vvvvvvvvvvvvvWw>~~~~~~~'
print_hl(wave_train)
center_print('')
program_title = 'SXSDLTOOL (BETA)'
print( center_space(program_title) + bold(program_title) )
center_print('A Python tool for the automated downloading of publicly available graviational wave data from the SXS Collaboration.')
author_info = 'Author: Koala Bear, Email: koalascript@gmail.com'
print( center_space(author_info)+bold(author_info) )
center_print('About SXS: black-holes.org')
center_print('')
print_hl(wave_train)
print('')
time.sleep(1)

# Print settings to screen
print_hl()
print( '>> '+bold('Settings file loaded:') )
print_hl()
settings.show()
time.sleep(4)

# --------------------------------------------------------------------------------- #
# For all SXS simulations available, download the metadatafiles into a special folder
# --------------------------------------------------------------------------------- #

# Get user's home directory
from os.path import expanduser
home = expanduser("~")

# # Name location where metadafiles will be stored
# work_dir = home+'/WORK/'
# mkdir(work_dir) # Note, this is a custom mkdir function
# # High level data storage directory
# data_dir = work_dir+'Data/'
# mkdir(data_dir)
# # Parent directory for waveform sets
# wave_dir = data_dir+'Waveforms/'
# mkdir(wave_dir)

# Get base directory from settings file
wave_dir = expanduser( settings.BASE_DIRECTORY )
if wave_dir[-1] is not '/':
    wave_dir = wave_dir + '/'



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            msg='Could not create %s.' % path
            raise

# Make sure that the otput directory exists
if not os.path.isfile(wave_dir) :
    mkdir_p(wave_dir)

# Location where waveforms and metadata will be stores
sxs_dir = wave_dir+'SXS/'
mkdir(sxs_dir)
# Location of Catalog file and temporary metadata
tmpdir = sxs_dir+'.mdstore/'
mkdir(tmpdir)

# --------------------------------------------------------------------------------- #
# For all SXS simulations available, download the metadatafiles into a special folder
# --------------------------------------------------------------------------------- #

# Alert the user that matadata dl is starting
print '>> '+bold('Starting the download of Meta Data.')

catalog_path = get_metadata(sxs_dir,tmpdir,verbose=True,replace_old=settings.REPLACE_OLD)

# Read the initial parameters from the meta data and test against desired inputs
search_results = sc_search( catalog_path, verbose=True, nonspinning=settings.NON_SPINNING )

# search_results = sc_search( verbose=True, nonspinning=settings.NON_SPINNING )

# --------------------------------------------------------------------------------- #
# For passing meta data, create the relevant simulation folder(s) and download the simlation h5 file structure
# --------------------------------------------------------------------------------- #

# Ensure that what to downlod is an iterable list
if type(settings.WHAT_TO_DOWNLOAD) is str:
    settings.WHAT_TO_DOWNLOAD = [settings.WHAT_TO_DOWNLOAD]

# Define the string format used for each file
for server_file in settings.WHAT_TO_DOWNLOAD:

    # Define the url location format that will be used for each download
    url_format = 'http://www.black-holes.org/waveforms/data/Download.php/?id=SXS:BBH:%04i&file=Lev%i/'+server_file+'.h5'

    # Define a prefix for data files
    data_prefix = server_file.split('_')[0] + '_'

    # Loop through resutls of sc_search
    for Y in search_results:

        # Get this run's ID
        run_id = Y.id

        # Get the simulation's resolution level -- this is the highest resolution by default
        Level = int( Y.simulation_name.split('Lev')[-1] )

        # Use the counter and Level varibale to create a url
        url = url_format % (run_id,Level)

        # Use the inctrement to uniquely name each file
        file_name = 'sxs%04i.h5' % run_id
        # file_name = 'sxs%04i.h5.tar' % run_id # Apr 2016: files are no longer tar'ed on the SXS server.

        # Only download if the decompressed file does not already exist
        run_dir = Y.source_dir[0]

        # Define smallest allowed file size
        min_bytes = 200

        # This is the expected content of the tar file. We only know this becuase of prior experience.
        # data_name = url_format.split('/')[-1]
        data_name = file_name

        #
        status_file = run_dir + '.status'
        download_state = smart_object()
        if not os.path.isfile(status_file):
            f = open( status_file, 'w' )
            f.write('\n')
            f.write('########################################################################\n')
            f.write('# 1=True if sxsdltool has finished with this folder, 0=False otherwise #\n')
            f.write('########################################################################\n')
            f.write('STATUS = 0\n')
            f.close()

        #
        download_state.learn_file(status_file)

        # Only proceed with download if this directory has not been passed before
        if not download_state.STATUS:

            # Determine whether the download is desired
            file_location = run_dir+data_name
            print('>> Querying "%s"' % file_location)
            DOWNLOAD = os.path.isfile(file_location) and settings.REPLACE_OLD
            DOWNLOAD = DOWNLOAD or not os.path.isfile(file_location)

            if DOWNLOAD:
                # Download the tar file
                print('>> Downloading TAR-FILE for:\t"%s"' % url )
                empty,tar_location = download(url,run_dir,save_name=file_name,\
                                            verbose=True, size_floor=min_bytes)

                '''Apr 2016: files are no longer tar'ed on the SXS server.'''
                # # Decompress
                # print('>> Decrompressing "%s"' % run_dir+file_name )
                # untar(tar_location,cleanup=True)
            else:
                # Let the user know that this download has been skipped
                print('>> '+bold('Skipping download. ')+'Data file already exists.')

            # Extract desired information form h5 file, then remove everything else to preserve disk space
            h5_file_string = run_dir + data_name
            h5tofiles( h5_file_string, run_dir, file_filter = lambda s: l_test(s,settings.L_MAX), \
                        cleanup = settings.DELETE_H5_FILES, prefix = data_prefix )

            # Mark thisdirectory as complete
            pattern = "STATUS"
            substitute = "STATUS = 1"
            replace_line(status_file, pattern, substitute)

        else:

            #
            print(bold('>> Status found to be complete. Moving on.'))


# Remove temporary folder of metedata and catalog
from shutil import rmtree
rmtree(tmpdir)
