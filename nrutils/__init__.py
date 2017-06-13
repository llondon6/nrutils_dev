#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
KOALA: A toolkit for data analysis, maths, and general science funs.
~ ll2'14

* see maths functions and related here: https://svn.einsteintoolkit.org/pyGWAnalysis/trunk/

# --- A sketch for the package structure --------------

# General object class for NR waveforms:
# Users will be encouraged to use lists of nrwf objects
from nrwf import *

# class object simulation catalog entries
# users will be encouraged to use lists of scentry objects
# a "catalog" is a list of scentry objects
from catalog import *

# Import io functions for package
from io import *

# Import analysis functions
from analysis import *

# import functions for generating analytic waveforms
from generate import *

#
from plotting import *


'''

# # Import all python files within this folder
from os.path import dirname, basename, isdir, realpath
from commands import getstatusoutput as bash

#
import os
verbose = True
if not (  os.environ.get("koala_verbose",'') == '' or os.environ.get("koala_verbose",'').lower() == 'true'  ):
    verbose = False


# Search recurssively within the config's sim_dir for files matching the config's metadata_id
this_file = realpath(__file__)
print "The highest level init for nrutils is located at: %s" % this_file
if this_file[-1] == 'c': this_file = this_file[:-1]
cmd = 'find %s -maxdepth 2 -name "__init__.py"' % dirname(this_file)
status, output = bash(cmd)

# make a list of all packages within the directory which contains this file
dir_list = output.split(chr(10))
internal_packages = [ basename(dirname(p)) for p in dir_list if not (p == this_file) ]

# Throw error is internal_packages is empty
if len(internal_packages) == 0:
    msg = '(!!) Unable to automatically find internal packages. Please report this bug to the developers. (https://github.com/llondon6/nrutils_dev/tree/master/nrutils)'
    raise ValueError(msg)

# Store package settings (useful directories etc) to a settings field
# NOTE that the __pathsfile__ variable is no longer used in favor of automatic directory assignments based in install location
# __pathsfile__ = [dirname(realpath(__file__))+'/settings/paths.ini']

# The instal path will be stored, but not used here. NOTE that the paths.ini file is technically not
# needed as path *could* be defined automatically, relative to the
__installpath__ = dirname(realpath(__file__))

#
__all__ = internal_packages


# Import all modules from each package
if verbose: print '\n>> Initiating nrutils ...'

# Let the people know
if verbose:
    print "\n>> Sub-Packages to be imported:"
    for k in internal_packages:
        print '   -> %s' % k

# Some other notes
if verbose: print '>> Please note style conventions:\
                  \n   * lower case function/method/variable names\
                  \n   * no underscore in names unless there are repeated letters, or counfounded syllables\
                  \n   * information is implicitely in time domain unless explicitely stated.\
                  \n   * frequency domain information will start with "fd".\n'

if verbose: print '%s:\n' % __name__
for p in internal_packages:
    if verbose: print '  .%s: ' % p
    exec r'import %s' % p
    # exec 'from %s import *' % p

# Import select modules for high level access
from manipulate import nr2h5
from core.units import *
from core.nrsc import scsearch,scbuild,gwylm,gwf,lswfa,scentry,screconf
from nrutils.formula import *

#
if verbose: print ''
# Cleanup
del cmd, bash, p, dir_list, status, output, this_file, basename, dirname, isdir, realpath

# for d in modules:
#     print '\n|&) Importing: %s' % d
#     # __import__(d,locals(),globals())
#     # cmd = 'from %s import *'
#     cmd = 'from %s import *' % d
#     exec cmd
