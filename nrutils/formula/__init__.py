# Import all python files within this folder
from os.path import dirname, basename, isdir
from nrutils.core.basics import yellow,green
from nrutils import verbose
import glob

# list all py files within this directory
modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]

# Dynamically import all modules within this folder (namespace preserving)
for module in modules:
    # if verbose: print yellow('    nrutils.formula:')+' Found formula module "%s"' % green(module)
    exec( 'import nrutils.formula.%s' % module)

# Cleanup
del dirname, basename, isdir, modules, module, glob
