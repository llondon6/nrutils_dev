
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
from numpy import array,ones,pi
from matplotlib import pyplot
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

#
q = 1.2
S1 = array( [0.0,0.0,0.0] )
S2 = array( [0.0,0.0,0.0] )

# Make sure that pycbc is visible. I do this here by activating a virtual installation.
system('source ~/.virtual_enviroments/pycbc-nr/bin/activate')

# Compute strain using lalsimulation. The default approximant is PhenomD
y = lswfa(q=q,S1=S1,S2=S2)

#
print d

# #
# y.plot(domain='time')
# y.plot(domain='freq')
#
# pyplot.show()
