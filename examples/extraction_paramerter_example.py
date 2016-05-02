#
# This example shows how to load data (create gwfylm objects) with the extraction_parameter input. This example assumes the user has installed the Cardiff-UIB dataset.
#
# lionel.london@ligo.org
#

#
from matplotlib.pyplot import *
from os import system
system('clear')

#
from nrutils.core.nrsc import *

# Search for simulations
A = scsearch(keyword="base",unique=True,verbose=True)

# Load waveforms using different exraction parameters. NOTE that if the extraction parameter input is not give, then the extraction parameter in the appropriate configuration file will be used. This default value may have also been set independently in the current offcial method for configuring nrutils.
y = []; lev = [3,4,5]
for v in lev:
    y.append( gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True, extraction_parameter=v ) )

# Compare the two waveforms by plotting
figure(); clr=rgb( len(y) )
a = [];
for k in range(len(y)):
    # NOTE that the time series of each
    plot( y[k].ylm[0].t, y[k].ylm[0].amp, color=clr[k], label='lev%i'%lev[k] )

#
pylim( y[k].ylm[0].t, y[k].ylm[0].amp )
legend(frameon=False); show()
