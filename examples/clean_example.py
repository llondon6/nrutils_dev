'''
Example of different methods to clean the junk radiation from NR data.
~ spxll'16
'''

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

# # Build the simulation catalog using the cofig files
# scbuild()

# Search for simulations
# A = scsearch(institute='sxs',nonspinning=True,q=[1,10],verbose=True,unique=True)
A = scsearch(keyword="base_96",unique=True,verbose=True)
# A = scsearch(precessing=True,q=[1,1.5],verbose=True,unique=True)
# A = scsearch(keyword="base",unique=True,verbose=True)

# Convert a single simulation into a waveform object with desired multipoles
y = gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True )

# We will want to perform two independt operations on the waveform and then compare. So let's copy the original object.
z = y.copy()

# Clean the waveform's junk radiation using a smooth window
y.clean(method='window') # NOTE that this is equivalent to y.clean()

# Clean the waveform's junk radiation by cropping
z.clean(method='crop')

# # Plot time domain strain
z.plot(kind='strain',show=False)

# plot time domain psi4
y.plot(kind='strain',show=True)
