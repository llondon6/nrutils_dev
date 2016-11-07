# Import all python files within this folder
from os.path import dirname, basename, isdir
from nrutils import verbose,__pathsfile__
import glob

# list all py files within this directory
modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]

# Import all lowest level classes and functions to this package level (not namespace preserving)
# if verbose: print ''

#
if verbose: print '      .basics*'
from pn import *

# Make definitions in the pathsfile avaliable to this package. Note that these will be treated as global settings for the nrutils package.
settings = smart_object( __pathsfile__ )

# Dynamically import all modules within this folder (namespace preserving)
for module in modules:
    if verbose: print '      .%s' % module
    exec 'import %s' % module
