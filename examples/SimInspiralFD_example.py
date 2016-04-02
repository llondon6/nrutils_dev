

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi
from matplotlib import pyplot
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

#
q = 40
S1 = array( [0.0,0.0,0.0] )
S2 = array( [0.0,0.0,0.0] )

# Please NOTE that you must have pycbc installed for this to work.

# Compute strain using lalsimulation. The default approximant is PhenomD.
# SimInspiralTD is called internally.
y = lswfa(q=q,S1=S1,S2=S2,domain='freq')
# y = lswfa(q=q,S1=S1,S2=S2)

# #
# y.plot(domain='time')
# y.plot(domain='freq')
#
# pyplot.show()

#
# system('deactivate')
