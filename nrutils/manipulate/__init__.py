# Import all python files within this folder
from os.path import dirname, basename, isdir
from nrutils import verbose
import glob

# list all py files within this directory
modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]

# Dynamically import all modules within this folder (namespace preserving)
for module in modules:
    # if verbose: print '      .%s' % module
    exec( 'from nrutils.manipulate.%s import *' % module)

# Cleanup
# del dirname,basename,isdir,glob,verbose,modules,module
