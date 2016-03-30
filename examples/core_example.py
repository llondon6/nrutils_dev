

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
# A = scsearch(keyword='base',verbose=True,unique=True)
A = scsearch(institute='sxs',nonspinning=True,q=1,verbose=True,unique=True)
# A = scsearch(precessing=True,q=[1,1.5],verbose=True,unique=True)

print dir(A[0])

# Convert a single simulation into a waveform object with desired multipoles
y = gwylm( scentry_obj = A[0], lm=[2,2], clean=True, dt=0.4, verbose=True )

# # Plot the waveform in time and frequency domains
# ax1 = y.plot(domain='freq',kind='psi4')
# ax2 = y.plot(domain='time',kind='psi4')
#
# pyplot.show()
