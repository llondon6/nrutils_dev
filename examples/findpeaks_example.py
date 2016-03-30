
'''

Example script for intpdiff

spxll'16

'''

#
from os import system
system('clear')

#
from nrutils.core.basics import *
from matplotlib.pyplot import *
from numpy import *

#
t = linspace(0,5*pi,1e3)

#
# random.seed(300)
noise = 2e-1*random.rand(len(t))

#
y0 = cos(t*2) + noise

#
y1 = intrp_diff( t, y0 )
y2 = intrp_diff( t, y0, n = 2 )

#
pks,locs = findpeaks(y0,min_distance=10)

rlocs = findroots(y0)

#
figure()

plot( t, y0, color='0.8', linewidth=1 )
# plot( t, y1, 'r--' )
# plot( t,-y2, '--b' )
plot( t[locs], y0[locs], 'or', label = 'peaks' )
plot( t[rlocs], y0[rlocs], 'sk', label='roots' )

xlim( lim(t) )
pylim( t, y0 )

light_grey = np.array([float(248)/float(255)]*3)
lg = legend(frameon=True, scatterpoints=1, fancybox=True, loc='lower left')
rect = lg.get_frame()
rect.set_facecolor(light_grey)
rect.set_linewidth(0.0)
lg.get_frame().set_alpha(0.2)

show()
