# Import all python files within this folder
from os.path import dirname, basename, isdir
from nrutils import verbose,__installpath__
import glob

# list all py files within this directory
modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]

# Import all lowest level classes and functions to this package level (not namespace preserving)
# if verbose: print ''

#
if verbose: print '      .basics*'
from basics import *

# # Make definitions in the pathsfile avaliable to this package. Note that these will be treated as global settings for the nrutils package.
# settings = smart_object( __pathsfile__ )
settings = smart_object()
# Path where institute specific cinfiguration files are held
settings.config_path = __installpath__ + '/config/'
# Path where catalog database files are to be stored
settings.database_path = __installpath__ + '/database/'
# File extension associated with catalog database files
settings.database_ext = 'db'

# Sign convention to be used with multipole moments:
M_RELATIVE_SIGN_CONVENTION = 1 # 1: Psi4 multipoles of are forced to have time domain frequencies of the SAME sign as m
                               #-1: Psi4 multipoles of are forced to have time domain frequencies of the OPPOSITE sign as m
                               # NOTE that here instanteneous phase angle will allways be defined as +atan(coss/pluss) -- no manual minus sign will ever be added when handling Psi4. However a manual minus sign is added when calculating strain.

# Dynamically import all modules within this folder (namespace preserving)
for module in modules:
    if verbose: print '      .%s' % module
    exec 'import %s' % module
