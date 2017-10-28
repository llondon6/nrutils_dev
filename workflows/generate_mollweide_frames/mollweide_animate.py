'''
Make frames for an animation of mollweide + time domain plotting of gwylm objects
'''

# Import usefuls
import os
os.system('clear')
from nrutils import scsearch,gwylm
import matplotlib
matplotlib.use('agg')
from matplotlib.pyplot import *
from numpy import array
from os.path import expanduser,join
from positive import *

#--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
# Options
#--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#

#
lmax = 5

#
outdir = '~/Desktop/mollweide_frames/'
mkdir(outdir,verbose=True)

#
kind = 'strain'

#--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#
# Main Flow
#--~--~--~--~--~--~--~--~--~--~--~--~--~--~--~--#

# Find sim
alert('Finding a simulation ...',header=True)
A = scsearch(keyword='base_96')[0]
# A = scsearch(q=[10,20],keyword='hr',verbose=True,institute='gt')[0]

#
framedir = join(outdir,A.simname)+'/'
mkdir(framedir,verbose=True)

# Load data
alert('Loading simulation data',header=True)
y = gwylm(A,lmax=lmax,verbose=True,clean=True)

#
t = y.lm[2,2]['strain'].t
frame_times = t[ range(0,len(t),10) ]

# Use information from gwylm's waveform characerization algorithms to determine start and end times of tinterest
tmin = y.lm[2,2]['strain'].t[ y.preinspiral.left_index ]
tmax = y.lm[2,2]['strain'].t[ y.postringdown.left_index ]

# Use the times above to crop the frame_times
frame_times = frame_times[ (frame_times>=tmin) & (frame_times<=tmax) ]

# Measure the frame_times relative to the peak of |h22|
real_t_peak = y.lm[2,2]['strain'].intrp_t_amp_max
frame_times -= real_t_peak

#
alert('Looping over all frame times. There will be %s frames.'%(yellow(str(len(frame_times)))),header=True)
for k,t in enumerate(frame_times):

    #
    R,C = 6,3

    #
    fig = figure( figsize=3*array([C,1.0*R]) )

    #
    ax4 = subplot2grid( (R,C), (0, 0), colspan=C, rowspan=3, projection='mollweide' )
    ax1 = subplot2grid( (R,C), (3, 0), colspan=C)
    ax2 = subplot2grid( (R,C), (4, 0), colspan=C, sharex=ax1)
    ax3 = subplot2grid( (R,C), (5, 0), colspan=C, sharex=ax1)

    # Make mollweide plot -- NOTE that the time input is relative to the peak in h22
    _,real_time = y.mollweide_plot(time=t,ax=ax4,form='abs',kind=kind,colorbar_shrink=0.8,N=100)
    ax4.set_title('$l_{max} = %i$'%max([l for l,m in y.lm]),size=20)


    #
    view_width = 300
    t_left = max(tmin,t+real_t_peak-100)
    if (t_left+view_width) < tmax:
        t_right = t_left+view_width
    else:
        t_right = tmax
        t_left = t_right - view_width
    wf_ax,_ = y.lm[2,2][kind].plot(ax=[ax1,ax2,ax3],tlim=[ t_left,t_right])
    for a in wf_ax:
        sca( a ); axvline( real_time, linestyle='-', color='k' )

    #
    subplots_adjust(hspace = 0.1)

    #
    image_path = expanduser(framedir)+'mollweide_%04d.png'%k
    alert('* Saving frame %i/%i to : "%s"'%(k+1,len(frame_times),image_path)  )
    savefig( image_path )

    #
    close('all')

alert('Trying to make video from frames.',header=True)
os.system('ffmpeg -start_number 48 -r 12 -f image2 -i %s/mollweide_%s.png -vcodec libx264 -crf 25  -pix_fmt yuv420p %s/%s.mp4'%( framedir, '%04d', ourdir, y.simname ))
