

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

# Use the copy method to duplicate y
g = y.copy()

# Plot frequency domain strain and show all current plots
if not ( y is g ):
    msg = geen('The copy '+bold('was')+' successful.')
    alert(msg)
    g.plot(kind='strain',show=True,domain='freq')
else:
    msg = 'The copy was '+red(bold('not'))+' successful.'
    error(msg)
