
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
t = linspace(0,12*pi,25)

#
y0 = cos(t)

#
y1 = intrp_diff( t, y0 )
y2 = intrp_diff( t, y0, n = 2 )

#
rlocs = findroots(y0)

#
figure()

plot( t, y0, color='0.8', linewidth=5 )
plot( t, y1, 'r--' )
plot( t,-y2, '--b' )
plot( t[rlocs], y0[rlocs], 'sk', label='roots' )

xlim( lim(t) )

legend()

show()
