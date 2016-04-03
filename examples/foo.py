
# TODO:
#
# * interpolate fd wavforms to have the same DeltaF
# * Apply PCA to FD parts
# * Plot PC-coeffs against initial parameters
#

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi,linspace
from matplotlib import pyplot
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

# Make sure that pycbc is visible. I do this here by activating a virtual installation.
system('source ~/.virtual_enviroments/pycbc-nr/bin/activate')

#
q_range = linspace(1,15,12)
S1 = array( [0.0,0.0,0.0] )
S2 = array( [0.0,0.0,0.0] )

# Compute strain using lalsimulation. The default approximant is PhenomD.
# SimInspiralTD is called internally.
y = []
pyplot.figure()
ax = pyplot.subplot(1,1,1)
# ax.set_xscale('log', nonposx='clip')
# ax.set_yscale('log')
for k,q in enumerate(q_range):
    wf = lswfa(q=q,S1=S1,S2=S2)
    y.append( wf.fd_amp )
    pyplot.plot( wf.t, wf.amp )
    msg = 'wf.dt = %f, wf.df = %f, N = %i' % (wf.dt,wf.df,wf.n)
    alert(msg)

#
pyplot.show()
