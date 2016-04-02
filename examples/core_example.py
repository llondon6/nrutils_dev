

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi
from matplotlib import pyplot
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

# # Build the simulation catalog using the cofig files
# scbuild()

# Search for simulations
A = scsearch(institute='sxs',nonspinning=True,q=1,verbose=True,unique=True)
# A = scsearch(precessing=True,q=[1,1.5],verbose=True,unique=True)

# Convert a single simulation into a waveform object with desired multipoles
y = gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True )

# Plot time domain strain
y.plot(kind='strain')

# plot time domain psi4
y.plot(kind='psi4')

# plot frequency domain strain and show all current plots
y.plot(kind='strain',show=True,domain='freq')
