
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

def linenum():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# Class for basic print manipulation
class print_format:
   magenta = '\033[95m'
   cyan = '\033[96m'
   darkcyan = '\033[36m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bold = '\033[1m'
   grey = gray = '\033[1;30m'
   ul = '\033[4m'
   end = '\033[0m'
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
def cyan(string):
    return print_format.cyan + string + print_format.end
def darkcyan(string):
    return print_format.darkcyan + string + print_format.end
def textul(string):
    return print_format.underline + string + print_format.end

#
def parent(path):
    '''
    Simple wrapper for getting absolute parent directory
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
                alert('Could not load: %s'%red(file_location))
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
    from numpy import array

    # Columate input.
    z = x.reshape((x.size,))

    # Return min and max as list
    return array([min(z),max(z)])

# Determine whether numpy array is uniformly spaced
def isunispaced(x,tol=1e-6):

    # import usefull fun
    from numpy import diff

    # If t is not a numpy array, then let the people know.
    if not type(x).__name__=='ndarray':
        msg = '(!!) The first input must be a numpy array of 1 dimension.'

    # Return whether the input is uniformly spaced
    return max(diff(x,2))<tol

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
        fname = 'note'

    print '('+cyan(fname)+')>> '+msg

#
def warning(msg,fname=None):

    if fname is None:
        fname = 'warning'

    print '('+yellow(fname)+')>> '+msg

#
def error(msg,fname=None):

    if fname is None:
        fname = 'error'

    raise ValueError( '('+red(fname)+')!! '+msg )



# Usual find methods can be slow AND non-verbose about what's happening. This is one possible solution that at least lets the user know what's happening in an online fashion.
def rfind( path , pattern = None, verbose = False, ignore = None ):

    #
    import fnmatch
    import os
    # Create a string with the current process name
    thisfun = inspect.stack()[0][3]

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
    elif a<b:
        true_hann = twice_hann[ true_width: ]
    else:
        proceed = False

    # Proceed (or not) with tapering
    taper = ones( len(arr) )
    if proceed:
        # Make the taper
        taper[ :min(state) ] = 0*taper[ :min(state) ]
        taper[ min(state) : max(state) ] = true_hann

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
    a = (-1.0)**m
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
    if abs(Y.imag) == 0:
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
    d = wfarr[0,0]-wfarr[1,0]
    if abs(delta-d)/delta < 1e-6:
        proceed = False
    else:
        proceed = True

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
def pad_wfarr(wfarr,new_length):

    #
    from numpy import hstack,zeros,arange

    # Only pad if size of the array is to increase
    length = len(wfarr[:,0])
    proceed = length < new_length

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
        _wfarr[:,0] = dt * arange( 0, new_length )

        # Create the pads for the other columns
        left_pad = zeros( int(new_length-length)/2 )
        right_pad = zeros( new_length-length-len(left_pad) )

        # Pad the remaining columns
        for k in arange(1,wfarr.shape[1]):
            _wfarr[:,k] = hstack( [left_pad,wfarr[:,k],right_pad] )

    else:

        # Otherwise, do nothing.
        _wfarr = wfarr

        # Warn the user that nothing has happened.
        msg = 'The desired new length is <= the current array length (i.e. number of time domain points). Nothing will be padded.'
        warning( msg,fname='pad_wfarr'+cyan('@%i'%linenum()) )

    # Return padded array
    return _wfarr

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
