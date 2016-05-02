

#
from os import system
system('clear')

#
from nrutils.core.nrsc import *
from numpy import array,ones,pi,linspace,allclose,mod,dot
from matplotlib import pylab as plt
from matplotlib import animation

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#
A = scsearch(keyword="base",unique=True,verbose=True)
y = gwylm( scentry_obj = A[0], lm=[2,2], dt=0.4, verbose=True )

# This step isn't really needed, but it's instructive to see how easy it is to pad a collection of multipole waveforms.
y.pad( new_length = y.ylm[0].n + mod(y.ylm[0].n,2) )

# Extract what will be plotted
t = y.hlm[0].t
x = y.hlm[0].plus

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(); clr = rgb(1)
grey = 0.9*ones((3,))

# Setup the frst axes
ax1 = plt.subplot(2,1,1)
ax1.set_xlim( lim(t) )
plt.grid(color='0.95', linestyle='-')
orig, = ax1.plot( t, x, lw=2, color=grey )
line, = ax1.plot( [], [], lw=1, color=clr, alpha=0.9 )
ax1.set_xlabel('$t$')
ax1.set_ylabel('$h_+(t), \;\; h_+(t+t_0)$')
ax1.set_axisbelow(True)
plt.setp(ax1.get_xticklabels(), visible=False)

# Setup the second
ax2 = plt.subplot(2,1,2)
ax2.set_xlim( lim(t) )
plt.grid(color='0.95', linestyle='-')
orig2, = ax2.plot( t, 0*t, lw=2, color=grey )
line2, = ax2.plot( [], [], lw=1, color=1-clr, alpha=0.5 )
ax2.set_xlabel('$t$')
ax2.set_ylabel('$h_+(t) - h_+(t+t_0)$')
ax2.set_axisbelow(True)

# initialization function: plot the background of each frame
def init():
    orig.set_data(t,x)
    line.set_data([], [])
    orig2.set_data(t,0*t)
    line2.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    shift = 10*i # mod( i * 16, len(t) )
    y = tshift( t, x, shift )
    line.set_data(t, y)
    line2.set_data(t, y-x)
    ax1.set_title(r'$t_0 = %1.2f$'%(shift),family='serif',fontsize=12)
    ax2.set_ylim( ax1.get_ylim() )
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed; however this option may not be compatible with your matplotlib version.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(1.5*len(x)/20), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
import os
savepath=''.join([parent(os.path.realpath(__file__)),'tshift_example.mp4'])
anim.save(savepath, fps=16, extra_args=['-vcodec', 'libx264'])

# Show the animation
plt.show()
