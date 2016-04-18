
# TODO:
#
# * interpolate fd wavforms to have the same DeltaF
# * Apply PCA to FD parts
# * Plot PC-coeffs against initial parameters
#
#* Ensure that all gwylm objects are in the same frame:
#  ---> Add a stand alone function that rotates a gwylm object from one frame into another in the time domain
#  ---> Add formatting method to gwylm to put it in the J or L frame when w = w0
#  ---> Add formatting method to convert static frame gwylm to dynamic frame gwylm (i.e. corotating frame transformation)
#
# * Create feature-alignment modules for GW-ML tools:
#   ---> generalized inner-product definition (e.g. frequency whitening or generalized kernel)
#   ---> time and phase alignment via optimal inner-product
#   ---> time and phase alignment via reference time or reference frequency
#
# * Interface feature-alignment modules with standard PCA tools:
#  ---> Simulation Catalog PCA: scpca() -- maybe a class
#  ---> Simulation Catalog Model: scmodel() -- a class
#  ---> Iterative and Adaptive methods: sclearn() -- definitely a class
#

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi,linspace,allclose,vstack,dot
from matplotlib.pyplot import *

#
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA as KPCA

#
from os import system
system('clear')

# from nrutils import *
from nrutils.core.nrsc import *

# Make sure that pycbc is visible. I do this here by activating a virtual installation.
system('source ~/.virtual_enviroments/pycbc-nr/bin/activate')

# Set physical parameters
apx = 'IMRPhenomPv2'
q_range = linspace(1,10,10)
S1 = array( [0.0,0.0,0.0] )
S2 = array( [0.0,0.0,0.0] )

# Generate a list of gwf objects
y = [];
for q in q_range: y.append( lswfa(q=q,S1=S1,S2=S2,fmin_hz=25,apx=apx) )

# Find the longest waveform, and then pad all of the waveforms to the same length. Note that they already have the same time spacing.
maxlen = max( [ k.n for k in y ] )
for k in y: k.pad(new_length=maxlen)

# # Plot Frequency domain amplitudes
# figure()
# ax = subplot(1,1,1)
# ax.set_xscale('log', nonposx='clip')
# ax.set_yscale('log', nonposx='clip', nonposy='clip')
# f = y[0].f
# for k,wf in enumerate(y):
#     plot( 2*pi*wf.f, wf.fd_amp, color=rgb(1,shift=-pi/4+float(k)/len(q_range)))
#     # Check that frequencies are the same
#     if not allclose(f,wf.f):
#         msg = 'frequencies are not close!'
#         error(msg)
# ax.set_xlim( 2*pi*lim(wf.f) )
# show()

#
W = vstack( [ wf.fd_amp for wf in y ] )
X = (W.T - np.mean(W.T, 0)) / np.std(W.T, 0)
X = X.T
figure()
plot(np.mean(W.T,0),'-o')

pca = PCA( n_components = X.shape[-1] )
pca.fit(X)
pca_score = pca.explained_variance_ratio_
V = pca.components_
print V.shape

#
# print pca.a

#
figure()
ax = []
ax.append( subplot(1,2,1) )
plot( range(len(pca_score)), pca_score, '-o' )
ax.append( subplot(1,2,2) )
ax[-1].set_yscale('log', nonposx='clip', nonposy='clip')
ax[-1].set_xscale('log', nonposx='clip')
plot( 2*pi*y[0].f, abs(V[0,:]), '-' )
show()

#
# coeffs = dot(V,)

#############################################################

# # Compute strain using lalsimulation. The default approximant is PhenomPv2.
# # SimInspiralTD is called internally.
# y = []
# figure()
# ax = subplot(1,1,1)
# ax.set_xscale('log', nonposx='clip')
# ax.set_yscale('log', nonposx='clip', nonposy='clip')
# for k,q in enumerate(q_range):
#     # Generate a PhenomPv2 waveform
#     wf = lswfa(q=q,S1=S1,S2=S2,fmin_hz=25,apx=apx)
#     # Pad the waveform
#     wf.pad(new_length=11163)
#     # Add the desired waveform data to a list
#     y.append( wf.fd_amp )
#     # Plotting
#     plot( 2*pi*wf.f, wf.fd_amp, color=rgb(1,shift=-pi/4+float(k)/len(q_range)))
#     msg = 'wf.dt = %f, wf.df = %f, N = %i' % (wf.dt,wf.df,wf.n)
#     alert(msg)
# #
# ax.set_xlim( 2*pi*lim(wf.f) )
# show()
