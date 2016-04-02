
# TODO:
#
# 2. build interface to load lal approximants as gwf objects
# 3. build alignment function for two gwf objects
# 4. build version of sc_pca
# 5. build version of sc_model in 2D! -- 3D???
# 6. build version of modelgrid
# 7. build version of sc_learn
# 8. apply sc_learn to PhenomD (P?? SEOBNRv3???)
# 9. Write APS talk
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
q_range = linspace(1,20,10)
S1 = array( [0.0,0.0,0.0] )
S2 = array( [0.0,0.0,0.0] )

# Compute strain using lalsimulation. The default approximant is PhenomD.
# SimInspiralTD is called internally.
y = []
pyplot.figure()
for k,q in enumerate(q_range):
    wf = lswfa(q=q,S1=S1,S2=S2)
    y.append( wf.fd_amp )
    pyplot.plot( wf.t, wf.plus )
    msg = 'wf.dt = %f, wf.df = %f' % (wf.dt,wf.df)
    alert(msg)

#
pyplot.show()
