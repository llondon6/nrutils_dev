# Import all python files within this folder
from os.path import dirname, basename, isdir, realpath
from commands import getstatusoutput as bash
from nrutils import verbose,__pathsfile__
import glob

# # list all py files within this directory
# modules = [ basename(f)[:-3] for f in glob.glob(dirname(__file__)+"/*.py") if not ('__init__.py' in f) ]
#
# # Dynamically import all modules within this folder (namespace preserving)
# for module in modules:
#     if verbose: print '      .%s' % module
#     exec 'import %s' % module

# Search recurssively within the config's sim_dir for files matching the config's metadata_id
this_file = realpath(__file__)
if this_file[-1] == 'c': this_file = this_file[:-1]
cmd = 'find %s -maxdepth 2 -name "__init__.py"' % dirname(this_file)
status, output = bash(cmd)

# make a list of all packages within the directory which contains this file
dir_list = output.split(chr(10))
internal_packages = [ basename(dirname(p)) for p in dir_list if not (p == this_file) ]

#
__all__ = internal_packages
# Import all modules from each package
# if verbose: print '\t%s:\n' % __name__
for p in internal_packages:
    if verbose: print '    .%s: ' % p
    exec 'import %s' % p
    # exec 'from %s import *' % p
