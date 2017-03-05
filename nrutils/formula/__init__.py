# Import all python files within this folder
from os.path import dirname, basename, isdir
from nrutils.basics import *
import glob, kerr

# list all py files within this directory
modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]

#
verbose = True

# Dynamically import all modules within this folder (namespace preserving)
for module in modules:
    if verbose: print yellow('kerr##')+' Found formula module "%s"' % green(module)
    exec 'import %s' % module

# Cleanup
del dirname, basename, isdir, modules, module, f, glob, kerr
