
#
from nrutils.core.nrsc import *

import matplotlib as mpl
# print mpl.rcParams
mpl.use('macosx')

# Search for simulations
# A = scsearch(institute='sxs',nonspinning=True,q=[1,10],verbose=True,unique=True)
# A = scsearch(precessing=True,q=[1,1.5],verbose=True,unique=True)
A = scsearch(keyword="base",unique=True,verbose=True)

# Convert a single simulation into a waveform object with desired multipoles
y = gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True )

# plot time domain strain

# from matplotlib.pyplot import *
# plot( y.ylm[0].t, y.ylm[0].plus )
# show()

y.plot(kind='strain',show=True,domain='time')
