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
verbose = True

# Search recurssively within the config's sim_dir for files matching the config's metadata_id
this_file = realpath(__file__)
if this_file[-1] == 'c': this_file = this_file[:-1]
cmd = 'find %s -name "__init__.py" -depth 2' % dirname(this_file)
status, output = bash(cmd)

# make a list of all packages within the directory which contains this file
dir_list = output.split(chr(10))
internal_packages = [ basename(dirname(p)) for p in dir_list if not (p == this_file) ]

# Store package settings (useful directories etc) to a settings field
__pathsfile__ = [dirname(realpath(__file__))+'/settings/paths.ini']

#
__all__ = internal_packages
# Import all modules from each package
if verbose: print '\n>> Initiating nrutils.'
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

#
if verbose: print ''
# Cleanup
del internal_packages, cmd, bash, p, dir_list, status, output, this_file, basename, dirname, isdir, realpath

# for d in modules:
#     print '\n|&) Importing: %s' % d
#     # __import__(d,locals(),globals())
#     # cmd = 'from %s import *'
#     cmd = 'from %s import *' % d
#     exec cmd
