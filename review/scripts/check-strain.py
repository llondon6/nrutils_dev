'''
The goal of this script is to compare the output of nrutils' strain calculation
method to the output of an independent MATLAB code of the same method. For convinience,
ascii data for the MATLAB routine's output is saved within this repository.
-- lionel.london@ligo.org 2016 --
'''

# Import useful things
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi,loadtxt
from matplotlib.pyplot import *
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

#
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Search for simulations: Use the CFUIB high resolution base case
A = scsearch(keyword="base_96",unique=True,verbose=True)

# Convert a single simulation into a waveform object with desired multipoles
y = gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True )

# load and plot external data files
matlab_output_file_location = '/Users/book/JOKI/Libs/KOALA/nrutils_dev/review/data/CFUIB0029_l2m2_r140.asc'
matlab_strain = loadtxt( matlab_output_file_location )

#
plot( matlab_strain[:,0], matlab_strain[:,1], color='b', label='dakit (MATLAB)' )
plot( y.hlm[0].t, y.hlm[0].amp, '--g', label='nrutils (Python)'  )
xlabel(r'$t/M$'); ylabel(r'$|rMh(t)|$');
legend(frameon=False,loc=2)
show()
savefig( 'check-strain-cfuib0029.pdf' )

# # plot frequency domain strain and show all current plots
# y.plot(kind='strain',show=True,domain='freq')
