
# Import Libs
import os,shutil
import glob
import urllib2
import tarfile,sys
import time
import subprocess
import re
import inspect
import pickle
import numpy
import string
import random
import h5py
import copy

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# NOTE that uncommenting the line below may cause errors in OSX install relating to fonts
# rc('text', usetex=True)

def linenum():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Class for basic print manipulation
class print_format:
   magenta = '\033[1;35m'
   cyan = '\033[0;36m'
   darkcyan = '\033[0;36m'
   blue = '\033[0;34m'
   # green = '\033[0;32m'
   green = '\033[92m'
   yellow = '\033[0;33m'
   red = '\033[31m'
   bold = '\033[1m'
   grey = gray = '\033[1;30m'
   ul = '\033[4m'
   end = '\x1b[0m'
   hlb = '\033[5;30;42m'
   underline = '\033[4m'

# Function that uses the print_format class to make tag text for bold printing
def bold(string):
    return print_format.bold + string + print_format.end
def red(string):
    return print_format.red + string + print_format.end
def green(string):
    return print_format.green + string + print_format.end
def magenta(string):
    return print_format.magenta + string + print_format.end
def blue(string):
    return print_format.blue + string + print_format.end
def grey(string):
    return print_format.grey + string + print_format.end
def yellow(string):
    return print_format.yellow + string + print_format.end
def orange(string):
    return print_format.orange + string + print_format.end
def cyan(string):
    return print_format.cyan + string + print_format.end
def darkcyan(string):
    return print_format.darkcyan + string + print_format.end
def hlblack(string):
    return print_format.hlb + string + print_format.end
def textul(string):
    return print_format.underline + string + print_format.end

# Return name of calling function
def thisfun():
    import inspect
    return inspect.stack()[2][3]

#
def parent(path):
    '''
    Simple wrapper for getting absolute parent directory
    '''
    return os.path.abspath(os.path.join(path, os.pardir))+'/'


# Make "mkdir" function for directories
def mkdir(dir_,rm=False,verbose=False):
    # Import useful things
    import os
    import shutil
    # Expand user if needed
    dir_ = os.path.expanduser(dir_)
    # Delete the directory if desired and if it already exists
    if os.path.exists(dir_) and (rm is True):
        if verbose:
            alert('Directory at "%s" already exists %s.'%(magenta(dir_),red('and will be removed')),'mkdir')
        shutil.rmtree(dir_,ignore_errors=True)
    # Check for directory existence; make if needed.
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        if verbose:
            alert('Directory at "%s" does not yet exist %s.'%(magenta(dir_),green('and will be created')),'mkdir')
    # Return status
    return os.path.exists(dir_)


# Function that returns true if for string contains l assignment that is less than l_max
def l_test(string,l_max):
    '''
    Function that returns true if for string contains l assignment that is <= l_max:
    score = ltest('Ylm_l3_m4_stuff.asc',3)
          = True
    score = ltest('Ylm_l3_m4_stuff.asc',5)
          = True
    score = ltest('Ylm_l6_m4_stuff.asc',2)
          = False
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


# Bash emulator
def bash( cmd ):
    # Pass the command to the operating system
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    raw_output = process.communicate()[0]
    #
    return raw_output

# Rough grep equivalent using the subprocess module
def grep( flag, file_location, options=None, comment=None ):
    #
    if options is None: options = ''
    if comment is None: comment = []
    if not isinstance(comment,list): comment = [comment]
    # Create string for the system command
    cmd = "grep " + '"' + flag + '" ' + file_location + options
    # Pass the command to the operating system
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    raw_output = process.communicate()[0]
    # Split the raw output into a list whose elements are the file's lines
    output = raw_output.splitlines()
    # Mask the lines that are comments
    if comment:
        for commet in comment:
            if not isinstance(commet,str):
                raise TypeError('Hi there!! Comment input must be string or list of stings. :D ')
            # Masking in Python:
            mask = [line[0]!=commet for line in output]
            output = [output[k] for k in xrange(len(output)) if mask[k]]

    # Return the list of lines
    return output

# Simple function to determine whether or not a string is intended to be a
# number: can it be cast as float?
def isnumeric( s ):
    try:
        float(s)
        ans = True
    except:
        ans = False
    return ans

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

    def __init__(this,attrfile=None,id=None,overwrite=False,**kwargs):
        #
        this.valid = False
        this.source_file_path = []
        this.source_dir  = []

        #
        this.overwrite = overwrite
        if attrfile is not None:

            if isinstance( attrfile, list ):

                # Learn list of files
                for f in attrfile:
                    this.learn_file( f, **kwargs )

            elif isinstance( attrfile, str ):

                # Learn file
                this.learn_file( attrfile, **kwargs )
            else:

                msg = 'first input (attrfile key) must of list containing strings, or single string of file location'
                raise ValueError(msg)

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
        line_list = grep(eqls,file_location,**kwargs)
        for line in line_list:
            this.learn_string( line,eqls, **kwargs )
        # Learn file location
        this.source_file_path.append(file_location)
        # Learn location of parent folder
        this.source_dir.append( parent(file_location) )

    # Function for parsing single lines strings into class attributes and values
    def learn_string(this,string,eqls='=',comment=None,**kwargs):

        #
        from numpy import array,ndarray,append

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        # Look for verbose key
        keys = ('verbose','verb')
        VERB = parsin( keys, kwargs )
        if VERB:
            print '[%s]>> VERBOSE mode on.' % thisfun
            print 'Lines with %s will not be considered.' % comment

        # Get rid of partial line comments. NOTE that full line comments have been removed in grep
        done = False
        if comment is not None:
            if not isinstance(comment,list): comment = [comment]
            for c in comment:
                if not isinstance(c,str):
                    raise TypeError('Hi there!! Comment input must be string or list of stings. I found %s :D '%[c])
                for k in range( string.count(c) ):
                    h = string.find(c)
                    # Keep only text that comes before the comment marker
                    string = string[:h]

        # The string must be of the format "A eqls B", in which case the result is
        # that the field A is added to this object with the value B
        part = string.split(eqls)

        # Remove harmful and unneeded characters from the attribute side
        attr = part[0].replace('-','_')
        attr = attr.replace(' ','')
        attr = attr.replace('#','')

        # Detect space separated lists on the value side
        # NOTE that this will mean that 1,2,3,4 5 is treated as 1,2,3,4,5
        part[1] = (','.join( [ p for p in part[1].split(' ') if p ] )).replace(',,',',')

        if VERB: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))
        # if True: print( '   ** Trying to learn:\n \t\t[%s]=[%s]' % (attr,part[1]))

        # Correctly formatted lines will be parsed into exactly two parts
        if [2 == len(part)]:
            #
            value = []
            if part[1].split(','):
                is_number = True
                for val in part[1].split(','):
                    #
                    if  not isnumeric(val):   # IF
                        is_number = False
                        if VERB: print( '>> Learning character: %s' % val )
                        value.append( val )
                    else:                       # Else
                        if VERB: print( '>> Learning number: %s' % val)
                        if val:
                            # NOTE that the line below contains eval rather than float becuase we want our data collation process to preserve type
                            value.append( eval(val) )
                #
                if is_number:
                    value = array(value)
            else:
                value.append("none")
            #
            if 1==len(value):
                value = value[0]

            if this.overwrite is False:
                # If the attr does not already exist, then add it
                if not ( attr in this.__dict__.keys() ):
                    setattr( this, attr, value )
                else:
                    # If it's already a list, then append
                    if isinstance( getattr(this,attr), (list,ndarray) ):
                        setattr(  this, attr, list(getattr(this,attr))  )
                        setattr(  this, attr, getattr(this,attr)+[value]  )
                    else:
                        # If it's not already a list, then make it one
                        old_value = getattr(this,attr)
                        setattr( this, attr, [old_value,value] )

            else:
                setattr( this, attr, value )

        else:
            raise ValueError('Impoperly formatted input string.')


# Function for loading various file types into numerical array
def smart_load( file_location,        # absolute path location of file
                  verbose = None ):     # if true, let the people know

    #
    from os.path import isfile
    from numpy import array

    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    #
    status = isfile(file_location)
    if status:

        # Handle various file types
        if file_location.split('.')[-1] is 'gz':
            # Load from gz file
            import gzip
            with gzip.open(file_location, 'rb') as f:
                raw = f.read()
        else:
            # Load from ascii file
            try:
                raw = numpy.loadtxt( file_location, comments='#')
            except:
                alert('Could not load: %s'%red(file_location),thisfun)
                alert(red('None')+' will be output',thisfun)
                raw = None
                status = False

    else:

        # Create a string with the current process name
        thisfun = inspect.stack()[0][3]

        #
        alert('Could not find file: "%s". We will proceed, but %s will be returned.'%(yellow(file_location),red('None')),thisfun)
        raw = None

    #
    return raw,status

# Function to produce array of color vectors
def rgb( N,                     #
         offset     = None,     #
         speed      = None,     #
         plot       = False,    #
         shift      = None,     #
         jet        = False,     #
         reverse    = False,     #
         verbose    = None ):   #

    #
    from numpy import array,pi,sin,arange,linspace

    # If bad first intput, let the people know.
    if not isinstance( N, int ):
        msg = 'First input must be '+cyan('int')+'.'
        raise ValueError(msg)

    #
    if offset is None:
        offset = pi/4.0

    #
    if speed is None:
        speed = 2.0

    #
    if shift is None:
        shift = 0

    #
    if jet:
        offset = -pi/2.1
        shift = pi/2.0

    #
    if reverse:
        t_range = linspace(1,0,N)
    else:
        t_range = linspace(0,1,N)

    #
    r = array([ 1, 0, 0 ])
    g = array([ 0, 1, 0 ])
    b = array([ 0, 0, 1 ])

    #
    clr = []
    w = pi/2.0
    for t in t_range:

        #
        R = r*sin( w*t                + shift )
        G = g*sin( w*t*speed + offset + shift )
        B = b*sin( w*t + pi/2         + shift )

        #
        clr.append( abs(R+G+B) )

    #
    if 1 == N :
        clr = clr[0]

    #
    if plot:

        #
        from matplotlib import pyplot as p

        #
        fig = p.figure()
        fig.set_facecolor("white")

        #
        for k in range(N):
            p.plot( array([0,1]), (k+1.0)*array([1,1])/N, linewidth=20, color = clr[k] )

        #
        p.axis('equal')
        p.axis('off')

        #
        p.ylim([-1.0/N,1.0+1.0/N])
        p.show()

    #
    return array(clr)

# custome function for setting desirable ylimits
def pylim( x, y, axis='both', domain=None, symmetric=False, pad_y=0.1 ):

    #
    from matplotlib.pyplot import xlim, ylim
    from numpy import ones

    #
    if domain is None:
        mask = ones( x.shape, dtype=bool )
    else:
        mask = (x>=min(domain))*(x<=max(domain))

    #
    if axis == 'x' or axis == 'both':
        xlim( lim(x) )

    #
    if axis == 'y' or axis == 'both':
        limy = lim(y[mask]); dy = pad_y * ( limy[1]-limy[0] )
        if symmetric:
            ylim( [ -limy[-1]-dy , limy[-1]+dy ] )
        else:
            ylim( [ limy[0]-dy , limy[-1]+dy ] )

# Return the min and max limits of an 1D array
def lim(x):

    # Import useful bit
    from numpy import array,amin,amax

    # Columate input.
    z = x.reshape((x.size,))

    # Return min and max as list
    return array([amin(z),amax(z)])

# Determine whether numpy array is uniformly spaced
def isunispaced(x,tol=1e-5):

    # import usefull fun
    from numpy import diff,amax

    # If t is not a numpy array, then let the people know.
    if not type(x).__name__=='ndarray':
        msg = '(!!) The first input must be a numpy array of 1 dimension.'

    # Return whether the input is uniformly spaced
    return amax(diff(x,2))<tol

# Calculate rfequency domain (~1/t Hz) given time series array
def getfreq( t, shift=False ):

    #
    from numpy.fft import fftfreq
    from numpy import diff,allclose,mean

    # If t is not a numpy array, then let the people know.
    if not type(t).__name__=='ndarray':
        msg = '(!!) The first input must be a numpy array of 1 dimension.'

    # If nonuniform time steps are found, then let the people know.
    if not isunispaced(t):
        msg = '(!!) The time input (t) must be uniformly spaced.'
        raise ValueError(msg)

    #
    if shift:
        f = fftshift( fftfreq( len(t), mean(diff(t)) ) )
    else:
        f = fftfreq( len(t), mean(diff(t)) )

    #
    return f

# Low level function for fixed frequency integration (FFI)
def ffintegrate(t,y,w0,n=1):

    # This function is based upon 1006.1632v1 Eq 27

    #
    from numpy import array,allclose,ones,pi
    from numpy.fft import fft,ifft,fftfreq,fftshift
    from numpy import where

    # If x is not a numpy array, then let the people know.
    if not type(y).__name__=='ndarray':
        msg = '(!!) The second input must be a numpy array of 1 dimension.'

    # If nonuniform time steps are found, then let the people know.
    if not isunispaced(t):
        msg = '(!!) The time input (t) must be uniformly spaced.'
        raise ValueError(msg)

    # Define the lowest level main function which applies integration only once.
    def ffint(t_,y_,w0=None):

        # Note that the FFI method is applied in a DOUBLE SIDED way, under the assumpion tat w0 is posistive
        if w0<0: w0 = abs(w0);

        # Calculate the fft of the inuput data, x
        f = getfreq(t_) # NOTE that no fftshift is applied

        # Replace zero frequency values with very small number
        if (f==0).any :
            f[f==0] = 1e-9

        #
        w = f*2*pi

        # Find masks for positive an negative fixed frequency regions
        mask1 = where( (w>0) * (w<w0)  ) # Positive and less than w0
        mask2 = where( (w<0) * (w>-w0) ) # Negative and greater than -w0

        # Preparare fills for each region of value + and - w0
        fill1 =  w0 * ones( w[mask1].shape )
        fill2 = -w0 * ones( w[mask2].shape )

        # Apply fills to the frequency regions
        w[ mask1 ] = fill1; w[ mask2 ] = fill2

        # Take the FFT
        Y_ = fft(y_)

        # Calculate the frequency domain integrated vector
        Y_int = Y_ / (w*1j)

        # Inverse transorm, and make sure that the inverse is of the same nuerical type as what was input
        tol = 1e-8
        y_isreal = allclose(y_.imag,0,atol=tol)
        y_isimag = allclose(y_.real,0,atol=tol)
        if y_isreal:
            y_int = ifft( Y_int ).real
        elif y_isimag:
            y_int = ifft( Y_int ).imag
        else:
            y_int = ifft( Y_int )

        # Share knowledge with the people.
        return y_int


    #
    x = y
    for k in range(n):
        #
        x = ffint(t,x,w0)

    #
    return x

#
def alert(msg,fname=None):

    if fname is None:
        fname = thisfun()

    print '('+cyan(fname)+')>> '+msg

#
def warning(msg,fname=None):

    if fname is None:
        fname = thisfun()

    print '('+yellow(fname+'!')+')>> '+msg

#
def error(msg,fname=None):

    if fname is None:
        fname = thisfun()

    raise ValueError( '('+red(fname+'!!')+')>> '+msg )



# Usual find methods can be slow AND non-verbose about what's happening. This is one possible solution that at least lets the user know what's happening in an online fashion.
def rfind( path , pattern = None, verbose = False, ignore = None ):

    #
    import fnmatch
    import os
    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

    # # Use find with regex to get matches
    # from subprocess import Popen, PIPE
    # (stdout, stderr) = Popen(['find',path,'-regex','.*/[^/]*%s*'%(pattern)], stdout=PIPE).communicate()
    #
    # if 'None' is stderr:
    #     raise ValueError( 'Unable to find files matching '+red(pattern)+' in '+red(path)+'. The system says '+red(stderr) )
    #
    # #
    # matches = stdout.split('\n')


    # All items containing these string will be ignored
    if ignore is None:
        ignore = ['.git','.svn']

    # Searching for pattern files. Let the people know.
    msg = 'Seaching for %s in %s:' % (cyan(pattern),cyan(path))
    if verbose: alert(msg,thisfun)

    matches = []
    for root, dirnames, filenames in os.walk( path ):
        for filename in filenames:

            proceed = len(filename)>=len(pattern)
            for k in ignore: proceed = proceed and not (k in filename)

            if proceed:

                if pattern in filename:
                    parts = os.path.join(root, filename).split(pattern)
                    if len(parts)==2:
                        if verbose: print magenta('  ->  '+parts[0])+cyan(pattern)+magenta(parts[1])
                    else:
                        if verbose: print magenta('  ->  '+os.path.join(root, filename) )
                    matches.append(os.path.join(root, filename))

    return matches


# Derivative function that preserves array length: [(d/dt)^n y(t)] is returned
def intrp_diff( t,        # domain values
                y,        # range values
                n = 1 ):  # degree of derivative

    #
    from numpy import diff,append
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    if 1 == n :
        #
        dt = t[1]-t[0]
        dy = diff(y)/dt
        dy_left  = append( dy, spline( t[:-1], dy )(t[-1]) )
        dy_right = append( spline( t[:-1], dy )(t[0]-dt), dy )
        dy_center = 0.5 * ( dy_left + dy_right )
        return dy_center
    elif n > 1:
        #
        dy = intrp_diff( t, y )
        return intrp_diff( t, dy, n-1 )
    elif n == 0 :
        #
        return y


# Find peaks adaptation from Matlab. Yet another example recursion's power!
def findpeaks( y, min_distance = None ):

    # Algorithm copied from Matlab's findLocalMaxima within findpeaks.m
    # lionel.london@ligo.org

    #
    from numpy import array,ones,append,arange,inf,isfinite,diff,sign,ndarray,hstack,where,abs
    import warnings

    #
    thisfun = inspect.stack()[0][3]

    if min_distance is None:

        #
        if not isinstance(y,ndarray):
            msg = red('Input must be numpy array')
            error(msg,thisfun)

        # bookend Y by NaN and make index vector
        yTemp = hstack( [ inf, y, inf ] )
        iTemp = arange( len(yTemp) )

        # keep only the first of any adjacent pairs of equal values (including NaN).
        yFinite = isfinite(yTemp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iNeq = where(  ( abs(yTemp[1:]-yTemp[:-1])>1e-12 )  *  ( yFinite[:-1]+yFinite[1:] )  )
        iTemp = iTemp[ iNeq ]

        # take the sign of the first sample derivative
        s = sign( diff(  yTemp[iTemp]  ) )

        # find local maxima
        iMax = where(diff(s)<0)

        # find all transitions from rising to falling or to NaN
        iAny = 1 + array( where( s[:-1]!=s[1:] ) )

        # index into the original index vector without the NaN bookend.
        iInflect = iTemp[iAny]-1
        iPk = iTemp[iMax]

        # NOTE that all inflection points are found, but note used here. The function may be updated in the future to make use of inflection points.

        # Package outputs
        locs    = iPk
        pks     = y[locs]

    else:

        #
        pks,locs = findpeaks(y)
        done = min( diff(locs) ) >= min_distance
        pks_ = pks
        c = 0
        while not done:

            #
            pks_,locs_ = findpeaks(pks_)
            print 'length is %i' % len(locs_)

            #
            if len( locs_ ) > 1 :
                #
                locs = locs[ locs_ ]
                pks = pks[ locs_ ]
                #
                done = min( diff(locs_) ) >= min_distance
            else:
                #
                done = True

            #
            c+=1
            print c

    #
    return pks,locs

# Find the roots of a descrete array.
def findroots( y ):

    from numpy import array,arange,allclose

    n = len(y)

    w =[]

    for k in range(n):
        #
        l = min(k+1,n-1)
        #
        if y[k]*y[l]<0 and abs(y[k]*y[l])>1e-12:
            #
            w.append(k)

        elif allclose(0,y[k],atol=1e-12) :
            #
            w.append(k)

    #
    root_mask = array( w )

    # #
    # _,root_mask = findpeaks( root_mask, min_distance=10 )

    #
    return root_mask

# Clone of MATLAB's find function: find all of the elements in a numpy array that satisfy a condition.
def find( bool_vec ):

    #
    from numpy import where

    #
    return where(bool_vec)[0]

# Low level function that takes in numpy 1d array, and index locations of start and end of wind, and then outputs the taper (a hanning taper). This function does not apply the taper to the data.
def maketaper(arr,state):

    # Import useful things
    from numpy import ones
    from numpy import hanning as hann

    # Parse taper state
    a = state[0]
    b = state[-1]

    # Only proceed if a valid taper is given
    proceed = True
    true_width = abs(b-a)
    twice_hann = hann( 2*true_width )
    if b>a:
        true_hann = twice_hann[ :true_width ]
    elif b<=a:
        true_hann = twice_hann[ true_width: ]
    else:
        proceed = False
        print a,b
        alert('Whatght!@!')

    # Proceed (or not) with tapering
    taper = ones( len(arr) )
    if proceed:
        # Make the taper
        if b>a:
            taper[ :min(state) ] = 0*taper[ :min(state) ]
            taper[ min(state) : max(state) ] = true_hann
        else:
            taper[ max(state): ] = 0*taper[ max(state): ]
            taper[ min(state) : max(state) ] = true_hann

    #
    if len(taper) != len(arr):
        error('the taper length is inconsistent with input array')

    #
    return taper


# James Healy 6/27/2012
# modifications by spxll'16
# conversion to python by spxll'16
def diff5( time, ff ):

    #
    from numpy import var,diff

    # check that time and func are the same size
    if length(time) != length(ff) :
        error('time and function arrays are not the same size.')

    # check that dt is fixed:
    if var(diff(time))<1e-8 :
        dt = time[1] - time[0]
        tindmax = len(time)
    else:
        error('Time step is not uniform.')

    # first order at the boundaries:
    deriv[1]         = ( -3.0*ff[4] + 16.0*ff[3] -36.0*ff[2] + 48.0*ff[1] - 25.0*ff[0] )/(12.0*dt)
    deriv[2]         = ( ff[5] - 6*ff[4] +18*ff[3] - 10*ff[2] - 3*ff[1] )/(12.0*dt)
    deriv[-2] = (  3.0*ff[-1] + 10.0*ff[-2] - 18*ff[-3] + 6*ff[-4] -   ff[-5])/(12.0*dt)
    deriv[-1]   = ( 25.0*ff[-1] - 48*ff[-2] + 36.0*ff[-3] -16*ff[-4] + 3*ff[-5])/(12.0*dt)

    # second order at interior:
    deriv[3:-2] = ( -ff[5:] + 8*ff[4:-1] - 8*ff[2:-3] + ff[1:-4] ) / (12.0*dt)

    #
    return deriv


# # Standard factorial function
# def factorial(n):
#     x = 1.0
#     for k in range(n):
#         x *= (k+1)
#     return x

# Simple combinatoric function -- number of ways to select k of n when order doesnt matter
def nchoosek(n,k): return factorial(n)/(factorial(k)*factorial(n-k))

#
# Use formula from wikipedia to calculate the harmonic
# See http://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics#Calculating
# for more information.
def sYlm(s,l,m,theta,phi):

    #
    from numpy import pi,ones,sin,tan,exp,array,double,sqrt,zeros
    from scipy.misc import factorial,comb

    #
    if isinstance(theta,(float,int,double)): theta = [theta]
    if isinstance(phi,(float,int,double)): phi = [phi]
    theta = array(theta)
    phi = array(phi)

    #
    theta = array([ double(k) for k in theta ])
    phi = array([ double(k) for k in phi ])

    # Ensure regular output (i.e. no nans)
    theta[theta==0.0] = 1e-9

    # Name anonymous functions for cleaner syntax
    f = lambda k: double(factorial(k))
    c = lambda x: double(comb(x[0],x[1]))
    cot = lambda x: 1.0/double(tan(x))

    # Pre-allocatetion array for calculation (see usage below)
    if min(theta.shape)!=1 and min(phi.shape)!=1:
        X = ones( len(theta) )
        if theta.shape != phi.shape:
            error('Input dim error: theta and phi inputs must be same size.')
    else:
        X = ones( theta.shape )


    # Calcualte the "pre-sum" part of sYlm
    a = (-1.0)**(m)
    a = a * sqrt( f(l+m)*f(l-m)*(2.0*l+1) )
    a = a / sqrt( 4.0*pi*f(l+s)*f(l-s) )
    a = a * sin( theta/2.0 )**(2.0*l)
    A = a * X

    # Calcualte the "sum" part of sYlm
    B = zeros(theta.shape)
    for k in range(len(theta)):
        B[k] = 0
        for r in range(l-s+1):
            if (r+s-m <= l+s) and (r+s-m>=0) :
                a = c([l-s,r])*c([l+s,r+s-m])
                a = a * (-1)**(l-r-s)
                a = a * cot( theta[k]/2.0 )**(2*r+s-m)
                B[k] = B[k] + a

    # Calculate final output array
    Y = A*B*exp( 1j*m*phi )

    #
    if sum(abs(Y.imag)) == 1e-7:
        Y = Y.real

    #
    return Y

# Interpolate waveform array to a given spacing in its first column
def intrp_wfarr(wfarr,delta=None,domain=None):

    #
    from numpy import linspace,array,diff,zeros,arange
    from scipy.interpolate import InterpolatedUnivariateSpline as spline

    # Validate inputs
    if (delta is None) and (domain is None):
        msg = red('First "delta" or "domain" must be given. See traceback above.')
        error(msg,'intrp_wfarr')
    if (delta is not None) and (domain is not None):
        msg = red('Either "delta" or "domain" must be given, not both. See traceback above.')
        error(msg,'intrp_wfarr')

    # Only interpolate if current delta is not input delta
    proceed = True
    if delta is not None:
        d = wfarr[0,0]-wfarr[1,0]
        if abs(delta-d)/delta < 1e-6:
            proceed = False

    # If there is need to interpolate, then interpolate.
    if proceed:

        # Encapsulate the input domain for ease of reference
        input_domain = wfarr[:,0]

        # Generate or parse the new domain
        if domain is None:
            N = diff(lim(input_domain))[0] / delta
            intrp_domain = delta * arange( 0, N  ) + wfarr[0,0]
        else:
            intrp_domain = domain

        # Pre-allocate the new wfarr
        _wfarr = zeros( (len(intrp_domain),wfarr.shape[1]) )

        # Store the new domain
        _wfarr[:,0] = intrp_domain

        # Interpolate the remaining columns
        for k in range(1,wfarr.shape[1]):
            _wfarr[:,k] = spline( input_domain, wfarr[:,k] )( intrp_domain )

    else:

        # Otherwise, return the input array
        _wfarr = wfarr

    #
    return _wfarr


# Fucntion to pad wfarr with zeros. NOTE that this should only be applied to a time domain waveform that already begins and ends with zeros.
def pad_wfarr(wfarr,new_length,where=None):

    #
    from numpy import hstack,zeros,arange

    # Only pad if size of the array is to increase
    length = len(wfarr[:,0])
    proceed = length < new_length

    #
    if isinstance(where,str):
        where = where.lower()

    #
    if where is None:
        where = 'sides'
    elif not isinstance(where,str):
        error('where must be string: left,right,sides','pad_wfarr')
    elif where not in ['left','right','sides']:
        error('where must be in {left,right,sides}','pad_wfarr')


    # Enforce integer new length
    if new_length != int(new_length):
        msg = 'Input pad length is not integer; I will apply int() before proceeding.'
        alert(msg,'pad_wfarr')
        new_length = int( new_length )

    #
    if proceed:


        # Pre-allocate the new array
        _wfarr = zeros(( new_length, wfarr.shape[1] ))

        # Create the new time series
        dt = wfarr[1,0] - wfarr[0,0]
        _wfarr[:,0] = dt * arange( 0, new_length ) + wfarr[0,0]

        if where is 'sides':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length)/2 )
            right_pad = zeros( new_length-length-len(left_pad) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k],right_pad] )

        elif where == 'right':

            # Create the pads for the other columns
            right_pad = zeros( new_length-length )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [wfarr[:,k],right_pad] )

        elif where == 'left':

            # Create the pads for the other columns
            left_pad = zeros( int(new_length-length) )

            # Pad the remaining columns
            for k in arange(1,wfarr.shape[1]):
                _wfarr[:,k] = hstack( [left_pad,wfarr[:,k]] )

    else:

        # Otherwise, do nothing.
        _wfarr = wfarr

        # Warn the user that nothing has happened.
        msg = 'The desired new length is <= the current array length (i.e. number of time domain points). Nothing will be padded.'
        warning( msg,fname='pad_wfarr'+cyan('@%i'%linenum()) )

    # Return padded array
    return _wfarr

# Shift a waveform arra by some "shift" amount in time
def tshift_wfarr( _wfarr, shift ):
    '''Shift a waveform arra by some "shift" amount in time'''
    # Import useful things
    from numpy import array
    # Unpack waveform array
    t,p,c = _wfarr[:,0],_wfarr[:,1],_wfarr[:,2]
    _y = p + 1j*c
    # Shift the waveform array data using tshift
    y = tshift( t,_y,shift )
    # Repack the input array
    wfarr = array(_wfarr)
    wfarr[:,0] = t
    wfarr[:,1] = y.real
    wfarr[:,2] = y.imag
    # Return answer
    ans = wfarr
    return ans


# Time shift array data, h, using a frequency diomain method
def tshift( t,      # time sries of data
            h,      # data that will be shifted
            t0 ):   # amount to shift data


    #
    from scipy.fftpack import fft, fftfreq, fftshift, ifft
    from numpy import diff,mean,exp,pi

    #
    is_real = sum( h.imag ) == 0

    # take fft of input
    H = fft(h)

    # get frequency domain of H in hertz (non-monotonic,
    # i.e. not the same as the "getfrequencyhz" function)
    dt = mean(diff(t))
    f = fftfreq( len(t), dt )

    # shift, and calculate ifft
    H_ = H * exp( -2*pi*1j*t0*f )

    #
    if is_real:
        h_ = ifft( H_ ).real
    else:
        h_ = ifft( H_ ) # ** here, errors in ifft process are ignored **

    #
    return h_

#
def pnw0(m1,m2,D=10.0):
    # https://arxiv.org/pdf/1310.1528v4.pdf
    # Equation 228
    # 2nd Reference: arxiv:0710.0614v1
    # NOTE: this outputs orbital frequency
    from numpy import sqrt,zeros,pi,array,sum
    #
    G = 1.0
    c = 1.0
    r = float(D)
    M = float( m1+m2 )
    v = m1*m2/( M**2 )
    gamma = G*M/(r*c*c)     # Eqn. 225
    #
    trm = zeros((4,))
    #
    trm[0] = 1.0
    trm[1] = v - 3.0
    trm[2] = 6 + v*41.0/4.0 + v*v
    trm[3] = -10.0 + v*( -75707.0/840.0 + pi*pi*41.0/64.0 ) + 19.0*0.5*v*v + v*v*v
    #
    w0 = sqrt( (G*M/(r*r*r)) * sum( array([ term*(gamma**k) for k,term in enumerate(trm) ]) ) )

    #
    return w0


# Find the interpolated global max location of a data series
def intrp_argmax( y,
                  domain=None,
                  verbose=False ):

    #
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    from scipy.optimize import minimize
    from numpy import linspace,argmax

    #
    x = range(len(y)) if domain is None else domain

    #
    yspline = spline( x, y )

    # Find the approximate max location in index
    k = argmax( y )

    # NOTE that we use minimize with bounds as it was found to have better behavior than fmin with no bounding
    x0 = x[k]
    f = lambda X: -yspline(X)
    q = minimize(f,x0,bounds=[(x0-10,x0+10)])
    xmax = q.x[0]

    #
    ans = xmax

    #
    return ans

# Shift phase of waveform array
def shift_wfarr_phase(wfarr,dphi):

    #
    from numpy import array,ndarray,sin,cos

    #
    if not isinstance(wfarr,ndarray):
        error( 'input must be numpy array type' )

    #
    t,r,c = wfarr[:,0],wfarr[:,1],wfarr[:,2]

    #
    r_ = r*cos(dphi) - c*sin(dphi)
    c_ = r*sin(dphi) + c*cos(dphi)

    #
    wfarr[:,0],wfarr[:,1],wfarr[:,2] = t , r_, c_

    #
    return wfarr

# Find the average phase difference and align two wfarr's
def align_wfarr_average_phase(this,that,mask=None,verbose=False):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    #
    from numpy import angle,unwrap,mean

    #
    if mask is None:
        u = this[:,1]+1j*this[:,2]
        v = that[:,1]+1j*that[:,2]
    else:
        u = this[mask,1]+1j*this[mask,2]
        v = that[mask,1]+1j*that[mask,2]

    #
    _a = unwrap( angle(u) )
    _b = unwrap( angle(v) )


    #
    a,b = mean( _a ), mean( _b )
    dphi = -a + b

    #
    if verbose:
        alert('The phase shift applied is %s radians.'%magenta('%1.4e'%(dphi)))

    #
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


#
def get_wfarr_relative_phase(this,that):

    #
    from numpy import angle,unwrap,mean

    #
    u = this[:,1]+1j*this[:,2]
    v = that[:,1]+1j*that[:,2]

    #
    _a = unwrap( angle(u) )[0]
    _b = unwrap( angle(v) )[0]

    #
    dphi = -_a + _b

    #
    return dphi

# Find the average phase difference and align two wfarr's
def align_wfarr_initial_phase(this,that):
    '''
    'this' phase will be aligned to 'that' phase over their domains
    '''

    dphi = get_wfarr_relative_phase(this,that)

    #
    this_ = shift_wfarr_phase(this,dphi)

    #
    return this_


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Here are some phenomenological fits used in PhenomD                               #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
# s is defined around Equation 3.6.
''' Copied from LALSimulation Version '''
def FinalSpin0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    s2 = s*s
    s3 = s2*s
    s4 = s3*s
    return 3.4641016151377544*eta - 4.399247300629289*eta2 +\
    9.397292189321194*eta3 - 13.180949901606242*eta4 +\
    (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +\
    (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +\
    (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +\
    (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4

#Wrapper function for FinalSpin0815_s.
''' Copied from LALSimulation Version '''
def FinalSpin0815(eta,chi1,chi2):
    from numpy import sqrt
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(abs(1.0 - 4.0*float(eta)))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # s defined around Equation 3.6 arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2)
    return FinalSpin0815_s(eta, s)

# Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
# Input parameter s defined around Equation 3.7 and 3.8.
def EradRational0815_s(eta,s):
    eta = round(eta,8)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta3*eta
    return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*\
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 \
    - 1.4756929437702908*eta + 7.304676214885011*eta2)*s)


# Wrapper function for EradRational0815_s.
def EradRational0815(eta, chi1, chi2):
    from numpy import sqrt,round
    eta = round(eta,8)
    if eta>0.25:
        error('symmetric mass ratio greater than 0.25 input')
    # Convention m1 >= m2
    Seta = sqrt(1.0 - 4.0*eta)
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    m1s = m1*m1
    m2s = m2*m2
    # arXiv:1508.07250
    s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)
    return EradRational0815_s(eta,s)


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#
# Given a 1D array, determine the set of N lines that are optimally representative  #
#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#

# Hey, here's a function that approximates any 1d curve as a series of lines
def romline(  domain,           # Domain of Map
              range_,           # Range of Map
              N,                # Number of Lines to keep for final linear interpolator
              positive=True,   # Toggle to use positive greedy algorithm ( where rom points are added rather than removed )
              verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_
    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/( R1 if abs(R1)!=0 else 1 )

    #
    if not positive:
        #
        done = False
        space = range( len(d) )
        raw_space = range( len(d) )
        err = lambda x: mean( abs(x) ) # std(x) #
        raw_mask = []
        while not done:
            #
            min_sigma = inf
            for k in range(len(space)):
                # Remove a trial domain point
                trial_space = list(space)
                trial_space.pop(k)
                # Determine the residual error incured by removing this trial point after linear interpolation
                # Apply linear interpolation ON the new domain TO the original domain
                trial_domain = d[ trial_space ]
                trial_range = r[ trial_space ]
                # Calculate the ROM's representation error using ONLY the points that differ from the raw domain, as all other points are perfectly represented by construction. NOTE that doing this significantly speeds up the algorithm.
                trial_mask = list( raw_mask ).append( k )
                sigma = err( linterp( trial_domain, trial_range )( d[trial_mask] ) - r[trial_mask] ) / ( err(r[trial_mask]) if err(r[trial_mask])!=0 else 1e-8  )
                #
                if sigma < min_sigma:
                    min_k = k
                    min_sigma = sigma
                    min_space = array( trial_space )

            #
            raw_mask.append( min_k )
            #
            space = list(min_space)

            #
            done = len(space) == N

        #
        rom = linterp( d[min_space], R[min_space] )
        knots = min_space

    else:
        from numpy import inf,argmin,argmax
        seed_list = [ 0, argmax(R), argmin(R), len(R)-1 ]
        min_sigma = inf
        for k in seed_list:
            trial_knots,trial_rom,trial_sigma = positive_romline( d, R, N, seed = k )
            # print trial_sigma
            if trial_sigma < min_sigma:
                knots,rom,min_sigma = trial_knots,trial_rom,trial_sigma

    #
    # print min_sigma

    return knots,rom


# Hey, here's a function related to romline
def positive_romline(   domain,           # Domain of Map
                        range_,           # Range of Map
                        N,                # Number of Lines to keep for final linear interpolator
                        seed = None,      # First point in domain (index) to use
                        verbose = False ):

    # Use a linear interpolator, and a reverse greedy process
    from numpy import interp, linspace, array, inf, arange, mean, zeros, std, argmax, argmin, amin, amax, ones
    linterp = lambda x,y: lambda newx: interp(newx,x,y)

    # Domain and range shorthand
    d = domain
    R = range_

    # Some basic validation
    if len(d) != len(R):
        raise(ValueError,'length of domain (of len %i) and range (of len %i) mus be equal'%(len(d),len(R)))
    if len(d)<3:
        raise(ValueError,'domain length is less than 3. it must be longer for a romline porcess to apply. domain is %s'%domain)

    # Normalize Data
    R0,R1 = mean(R), std(R)
    r = (R-R0)/R1
    #
    weights = (r-amin(r)) / amax( r-amin(r) )
    weights = ones( d.size )

    #
    if seed is None:
        seed = argmax(r)
    else:
        if not isinstance(seed,int):
            msg = 'seed input must be int'
            error( msg, 'positive_romline' )

    #
    done = False
    space = [ seed ]
    domain_space = range(len(d))
    err = lambda x: mean( abs(x) ) # std(x) #
    min_space = list(space)
    while not done:
        #
        min_sigma = inf
        for k in [ a for a in domain_space if not (a in space) ]:
            # Add a trial point
            trial_space = list(space)
            trial_space.append(k)
            trial_space.sort()
            # Apply linear interpolation ON the new domain TO the original domain
            trial_domain = d[ trial_space ]
            trial_range = r[ trial_space ]
            #
            sigma = err( weights * (linterp( trial_domain, trial_range )( d ) - r) ) / ( err(r) if err(r)!=0 else 1e-8  )
            #
            if sigma < min_sigma:
                min_k = k
                min_sigma = sigma
                min_space = array( trial_space )

        #
        space = list(min_space)
        #
        done = len(space) == N

    #
    rom = linterp( d[min_space], R[min_space] )
    knots = min_space

    return knots,rom,min_sigma


# Fix nans, nonmonotinicities and jumps in time series waveform array
def straighten_wfarr( wfarr, verbose=False ):
    '''
    Some waveform arrays (e.g. from the BAM code) may have non-monotonic time series
    (gaps, duplicates, and crazy backwards referencing). This method seeks to identify
    these instances and reformat the related array. Non finite values will also be
    removed.
    '''

    # Import useful things
    from numpy import arange,sum,array,diff,isfinite,hstack
    thisfun = 'straighten_wfarr'

    # Remove rows that contain non-finite data
    finite_mask = isfinite( sum( wfarr, 1 ) )
    if sum(finite_mask)!=len(finite_mask):
        if verbose: alert('Non-finite values found in waveform array. Corresponding rows will be removed.',thisfun)
    wfarr = wfarr[ finite_mask, : ]

    # Sort rows by the time series' values
    time = array( wfarr[:,0] )
    space = arange( wfarr.shape[0] )
    chart = sorted( space, key = lambda k: time[k] )
    if (space != chart).all():
        if verbose: alert('The waveform array was found to have nonmonotinicities in its time series. The array will now be straightened.',thisfun)
    wfarr = wfarr[ chart, : ]

    # Remove rows with duplicate time series values
    time = array( wfarr[:,0] )
    diff_mask = hstack( [ True, diff(time).astype(bool) ] )
    if sum(diff_mask)!=len(diff_mask):
        if verbose: alert('Repeated time values were found in the array. Offending rows will be removed.',thisfun)
    wfarr = wfarr[ diff_mask, : ]

    # The wfarr should now be straight
    # NOTE that the return here is optional as all operations act on the original input
    return wfarr


#
def rISCO_14067295(a):
    """
    Calculate the ISCO radius of a Kerr BH as a function of the Kerr parameter using eqns. 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)

    Parameters
    ----------
    a : Kerr parameter

    Returns
    -------
    ISCO radius
    """

    import numpy as np
    a = np.array(a)

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.+(1.-a**2.)**(1./3)*((1.+a)**(1./3) + (1.-a)**(1./3))
    z2 = np.sqrt(3.*a**2 + z1**2)
    a_sign = np.sign(a)
    return 3+z2 - np.sqrt((3.-z1)*(3.+z1+2.*z2))*a_sign

# https://arxiv.org/pdf/1406.7295.pdf
def Mf14067295( m1,m2,chi1,chi2,chif=None ):

    import numpy as np

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m
    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    #
    if chif is None:
        chif = jf14067295(m1, m2, chi1, chi2)
    r_isco = rISCO_14067295(chif)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    M0  = 0.951507
    K1  = -0.051379
    K2a = -0.004804
    K2b = -0.054522
    K2c = -0.000022
    K2d = 1.995246
    K3a = 0.007064
    K3b = -0.017599
    K3c = -0.119175
    K3d = 0.025000
    K4a = -0.068981
    K4b = -0.011383
    K4c = -0.002284
    K4d = -0.165658
    K4e = 0.019403
    K4f = 2.980990
    K4g = 0.020250
    K4h = -0.004091
    K4i = 0.078441

    # binding energy at ISCO -- Eq.(2.7) of Ori, Thorne Phys Rev D 62 124022 (2000)
    E_isco = (1. - 2./r_isco + chif/r_isco**1.5)/np.sqrt(1. - 3./r_isco + 2.*chif/r_isco**1.5)

    # final mass -- Eq. (14) of Healy et al Phys Rev D 90, 104004 (2014)
    mf = (4.*eta)**2*(M0 + K1*S + K2a*Delta*delta_m + K2b*S**2 + K2c*Delta**2 + K2d*delta_m**2 \
        + K3a*Delta*S*delta_m + K3b*S*Delta**2 + K3c*S**3 + K3d*S*delta_m**2 \
        + K4a*Delta*S**2*delta_m + K4b*Delta**3*delta_m + K4c*Delta**4 + K4d*S**4 \
        + K4e*Delta**2*S**2 + K4f*delta_m**4 + K4g*Delta*delta_m**3 + K4h*Delta**2*delta_m**2 \
        + K4i*S**2*delta_m**2) + (1+eta*(E_isco+11.))*delta_m**6.

    return mf*m

#
def jf14067295_diff(a_f,eta,delta_m,S,Delta):
    """ Internal function: the final spin is determined by minimizing this function """

    #
    import numpy as np

    # calculate ISCO radius
    r_isco = rISCO_14067295(a_f)

    # angular momentum at ISCO -- Eq.(2.8) of Ori, Thorne Phys Rev D 62 124022 (2000)
    J_isco = (3*np.sqrt(r_isco)-2*a_f)*2./np.sqrt(3*r_isco)

    # fitting coefficients - Table XI of Healy et al Phys Rev D 90, 104004 (2014)
    # [fourth order fits]
    L0  = 0.686710
    L1  = 0.613247
    L2a = -0.145427
    L2b = -0.115689
    L2c = -0.005254
    L2d = 0.801838
    L3a = -0.073839
    L3b = 0.004759
    L3c = -0.078377
    L3d = 1.585809
    L4a = -0.003050
    L4b = -0.002968
    L4c = 0.004364
    L4d = -0.047204
    L4e = -0.053099
    L4f = 0.953458
    L4g = -0.067998
    L4h = 0.001629
    L4i = -0.066693

    a_f_new = (4.*eta)**2.*(L0  +  L1*S +  L2a*Delta*delta_m + L2b*S**2. + L2c*Delta**2 \
        + L2d*delta_m**2. + L3a*Delta*S*delta_m + L3b*S*Delta**2. + L3c*S**3. \
        + L3d*S*delta_m**2. + L4a*Delta*S**2*delta_m + L4b*Delta**3.*delta_m \
        + L4c*Delta**4. + L4d*S**4. + L4e*Delta**2.*S**2. + L4f*delta_m**4 + L4g*Delta*delta_m**3. \
        + L4h*Delta**2.*delta_m**2. + L4i*S**2.*delta_m**2.) \
        + S*(1. + 8.*eta)*delta_m**4. + eta*J_isco*delta_m**6.

    daf = a_f-a_f_new
    return daf*daf

#
def jf14067295(m1, m2, chi1, chi2):
    """
    Calculate the spin of the final BH resulting from the merger of two black holes with non-precessing spins using fit from Healy et al Phys Rev D 90, 104004 (2014)

    Parameters
    ----------
    m1, m2 : component masses
    chi1, chi2 : dimensionless spins of two BHs

    Returns
    -------
    dimensionless final spin, chif
    """
    import numpy as np
    import scipy.optimize as so

    if np.any(abs(chi1>1)):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2>1)):
      raise ValueError("chi2 has to be in [-1, 1]")

    # Vectorize the function if arrays are provided as input
    if np.size(m1) * np.size(m2) * np.size(chi1) * np.size(chi2) > 1:
        return np.vectorize(bbh_final_spin_non_precessing_Healyetal)(m1, m2, chi1, chi2)

    # binary parameters
    m = m1+m2
    q = m1/m2
    eta = q/(1.+q)**2.
    delta_m = (m1-m2)/m

    S1 = chi1*m1**2 # spin angular momentum 1
    S2 = chi2*m2**2 # spin angular momentum 2
    S = (S1+S2)/m**2 # symmetric spin (dimensionless -- called \tilde{S} in the paper)
    Delta = (S2/m2-S1/m1)/m # antisymmetric spin (dimensionless -- called tilde{Delta} in the paper

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # compute the final spin
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    x, cov_x = so.leastsq(jf14067295_diff, 0., args=(eta, delta_m, S, Delta))
    chif = x[0]

    return chif


#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#
# Find the polarization and orbital phase shifts that maximize the real part
# of  gwylm object's (2,2) and (2,1) multipoles at merger (i.e. the sum)
''' See gwylm.selfalign for higher level Implementation '''
#00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%00%%#

def vectorize( _gwylmo, dphi, dpsi, k_ref=0 ):
    from numpy import array
    vec = []
    select_modes = [ (2,2), (2,1) ]
    valid_count = 0
    gwylmo = _gwylmo.rotate( dphi=dphi, dpsi=dpsi, apply=False, verbose=False, fast=True )
    for y in gwylmo.ylm:
        l,m = y.l,y.m
        if (l,m) in select_modes:
            vec.append( y.plus[ k_ref ] )
            valid_count += 1
    if valid_count != 2:
        error('input gwylm object must have both the l=m=2 and (l,m)=(2,1) multipoles; only %i of these was found'%valid_count)
    return array(vec)

def alphamax(_gwylmo,dphi,plt=False,verbose=False,n=13):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,sum,argmax,array
    action = lambda x: sum( vectorize( _gwylmo, x[0], x[1] ) )
    dpsi_range = linspace(-1,1,n)*pi
    dpsis = linspace(-1,1,1e2)*pi
    a = array( [ action([dphi,dpsi]) for dpsi in dpsi_range ] )
    aspl = spline( dpsi_range, a, kind='cubic' )(dpsis)
    dpsi_opt_guess = dpsis[argmax(aspl)]
    K = minimize( lambda PSI: -action([dphi,PSI]), dpsi_opt_guess )
    dpsi_opt = K.x[-1]
    if plt:
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot, xlabel
        plot( dpsi_range, a, linewidth=4, color='k', alpha=0.1 )
        plot( dpsis, aspl, label=dpsi )
        plot( dpsis[argmax(aspl)], aspl[argmax(aspl)], 'or', mfc='none' )
        xlabel(r'$\psi$')
    if verbose: print dpsi_opt,action([dphi,dpsi_opt])
    return [ dpsi_opt, action([dphi,dpsi_opt])    ]

def betamax(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array
    dphi_list = pi*linspace(-1,1,n)
    dpsi,val = [],[]
    for dphi in dphi_list:
        [dpsi_,val_] = alphamax(_gwylmo,dphi,plt=False,n=n)
        dpsi.append( dpsi_ )
        val.append( val_ )

    dphis = linspace(min(dphi_list),max(dphi_list),1e3)
    vals = spline( dphi_list, val, kind='cubic' )( dphis )
    dpsi_s = spline( dphi_list, dpsi, kind='cubic' )( dphis )

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )
    dphi_opt_guess = dphis[argmax(vals)]
    dpsi_opt_guess = dpsi_s[argmax(vals)]
    if opt:
        K = minimize( action, [dphi_opt_guess,dpsi_opt_guess] )
        # print K
        dphi_opt,dpsi_opt = K.x
        val_max = -K.fun
    else:
        dphi_opt = dphi_opt_guess
        dpsi_opt = dpsi_opt_guess
        val_max = vals.max()

    if plt:
        # Setup plotting backend
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import axes3d
        mpl.rcParams['lines.linewidth'] = 0.8
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 20
        from matplotlib.pyplot import plot,xlabel,title
        plot( dphi_list, val, linewidth=4, alpha=0.1, color='k' )
        plot( dphi_opt, val_max, 'or', alpha=0.5 )
        plot( dphis, vals )
        xlabel(r'$\phi$')
        title(val_max)

    if verbose:
        print 'dphi_opt = ' + str(dphi_opt)
        print 'dpsi_opt = ' + str(dpsi_opt)
        print 'val_max = ' + str(val_max)

    return dphi_opt,dpsi_opt

def betamax2(_gwylmo,n=10,plt=False,opt=True,verbose=False):
    from scipy.interpolate import interp1d as spline
    from scipy.optimize import minimize
    from numpy import pi,linspace,argmax,array

    action = lambda x: -sum( vectorize( _gwylmo, x[0], x[1] ) )

    dphi,dpsi,done,k = pi,pi/2,False,0
    while not done:
        dpsi_action = lambda _dpsi: action( [dphi,_dpsi] )
        dpsi = minimize( dpsi_action, dpsi, bounds=[(0,2*pi)] ).x[0]
        dphi_action = lambda _dphi: action( [_dphi,dpsi] )
        dphi = minimize( dphi_action, dphi, bounds=[(0,2*pi)] ).x[0]
        done = k>n
        print '>> ',dphi,dpsi,action([dphi,dpsi])
        k+=1

    return dphi,dpsi
