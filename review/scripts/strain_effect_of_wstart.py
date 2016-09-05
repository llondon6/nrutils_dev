

#
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi,linspace
from os import system
system('clear')

#
from nrutils import scbuild,scsearch,gwylm,rgb,pylim
from matplotlib.pyplot import plot, show, legend, axvline, xlim, ylim, savefig, xlabel, ylabel, title

# Search for simulations
A = scsearch(keyword="base_96",unique=True,verbose=True)[0]

#
l,m = 2,2

#
raw_w22 = A.raw_metadata.freq_start_22
delta = 0.2 * raw_w22
w22_range = linspace( raw_w22 - delta, raw_w22 + delta, 2 )

# Use a range of w22 values
y = []
for w22 in w22_range:
    y.append(  gwylm( scentry_obj = A, lm=[l,m], w22=w22, verbose=True )  )

# Use raw w22 from bbh metadata
x = gwylm( scentry_obj = A, lm=[l,m], w22=raw_w22, verbose=True )

# Use the nrutils internal algorithm to estimate w22
z = gwylm( scentry_obj = A, lm=[l,m], dt=0.4, verbose=True )

#
color = rgb( len(y) )
for h,k in enumerate(y):
    plot( k.hlm[0].t-x.hlm[0].t[0], k.hlm[0].amp, color = color[h], label=r'$\omega_{22} = %1.4f \mathrm{(other)}$' % w22_range[h] )

plot( x.hlm[0].t-x.hlm[0].t[0], x.hlm[0].amp, label=r'$\omega_{22} = %1.4f \mathrm{(bbh\;\;file)}$' % raw_w22 )
plot( z.hlm[0].t-x.hlm[0].t[0], z.hlm[0].amp, label=r'$\omega_{22} = %1.4f \mathrm{(nrutils)}$' % z.wstart_pn )

legend()
axvline( 140+140, color='r' )
# xlim([120,310])
# ylim([-0.14,0.45])
pylim( x.hlm[0].t-x.hlm[0].t[0], x.hlm[0].amp )
xlabel('$t/M$')
# ylabel(r'$\frac{d\phi(t)}{dt}$')
ylabel('$|h_{%i%i}(t)|$'%(l,m))
title(A.label)
savefig('strain_effect_of_wstart_amp.png')
show()
