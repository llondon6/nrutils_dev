
#
from nrutils.core.basics import *

#
class gwylm_radiation_axis_workflow:

    '''
    Workflow to calculate optimal emission axis (i.e. radiation axis), and related quantities. Comprehensive plotting is suggested and included as an option.
    '''

    #
    def __init__( this, gwylmo, kind=None, plot=False, outdir=None, save=False, safe_domain_range=None, verbose=True ):

        #
        from os.path import expanduser
        from numpy import isnan,array

        # Calculate radiated and remnant quantities
        alert('Calculating Radiated Quantities','gwylm_radiation_axis_workflow',verbose=verbose)
        gwylmo.__calc_radiated_quantities__(use_mask=False)

        # Store reference to input gwylmo
        this.gwylmo = gwylmo

        # NOTE that relevant information will be stored within the current object
        this.radiation_axis = {}
        rax = this.radiation_axis

        #
        if kind is None: kind = 'psi4'

        #
        this.save = save

        #
        if outdir is None: outdir = '~/Desktop/'+gwylmo.simname
        if save:
            outdir = expanduser( outdir ); this.outdir = outdir
            mkdir( this.outdir, verbose=verbose )

        # Calculate radiation axes in time and frequency domain
        alert('Calculating TD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
        td_alpha,td_beta,td_gamma,td_x,td_y,td_z,td_domain = this.calc_radiation_axis( domain = 'time', kind = kind, safe_domain_range=None  )
        alert('Calculating FD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
        fd_alpha,fd_beta,fd_gamma,fd_x,fd_y,fd_z,fd_domain = this.calc_radiation_axis( domain = 'freq', kind = kind, safe_domain_range=safe_domain_range  )

        # # Mask away Nans
        # mask = array([not isnan(v) for v in fd_beta+fd_alpha+fd_gamma])
        # print len(fd_beta)-sum(mask)
        # fd_alpha = fd_alpha[mask]
        # fd_beta  = fd_beta[mask]
        # fd_gamma = fd_gamma[mask]
        # fd_domain=fd_domain[mask]
        # fd_x = fd_x[mask]
        # fd_y = fd_y[mask]
        # fd_z = fd_z[mask]

        # Store time domain data
        rax['td_alpha'],rax['td_beta'],rax['td_gamma'] = td_alpha,td_beta,td_gamma
        rax['td_x'],rax['td_y'],rax['td_z'] = td_x,td_y,td_z
        rax['td_domain'] = td_domain

        # Store freq domain data
        rax['fd_alpha'],rax['fd_beta'],rax['fd_gamma'] = fd_alpha,fd_beta,fd_gamma
        rax['fd_x'],rax['fd_y'],rax['fd_z'] = fd_x,fd_y,fd_z
        rax['fd_domain'] = fd_domain

        # Define plotting function to be called either internally or externally
        def internal_plotting_function():

            #
            alert('Plotting TD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
            this.plot_radiation_axis_3panel( domain='time' )
            alert('Plotting FD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
            this.plot_radiation_axis_3panel( domain='freq' )

            #
            view = None
            this.plot_radiation_axis_on_sphere( domain='time', view=view )
            this.plot_radiation_axis_on_sphere( domain='freq', view=view )
            #
            view = (90,270)
            this.plot_radiation_axis_on_sphere( domain='time', view=view )
            this.plot_radiation_axis_on_sphere( domain='freq', view=view )

        # Store internal plotting funtion to this object
        this.plot = internal_plotting_function

        #
        if plot: this.plot()

    # Encapsulation of calc angles given domain and type
    def calc_radiation_axis( this, domain=None, kind = None, safe_domain_range=None ):

        # Calc radiation axis: alpha beta gamma and x y z
        kind = 'psi4' if kind is None else kind
        domain = 'time' if domain is None else domain

        #
        gwylmo = this.gwylmo

        # Construct dictionary of multipoles using all multipoles available
        mp = { (l,m) : ( gwylmo.lm[l,m][kind].y if domain in ('t','time') else gwylmo.lm[l,m][kind].fd_y ) for l,m in gwylmo.lm  }
        # Domain values: time or freq
        domain_vals = gwylmo.lm[2,2][kind].t if domain in ('t','time') else gwylmo.lm[2,2][kind].f

        # Calculate corotating angles using low-level function
        alpha,beta,gamma,x,y,z = calc_coprecessing_angles( mp, domain_vals, ref_orientation=gwylmo.J, return_xyz='all', safe_domain_range = ([0.01,0.1] if safe_domain_range is None else safe_domain_range) if domain in ('f','freq') else None )

        # return answers
        return alpha,beta,gamma,x,y,z,domain_vals

    #
    def plot_radiation_axis_3panel( this, domain=None, kind = None ):

        #
        from matplotlib.pyplot import figure, figaspect, plot, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, subplot, grid, title, draw, show, savefig
        from matplotlib.pyplot import close as close_figure
        from numpy import mod,pi,hstack,array,ones
        from os.path import join

        # Calc radiation axis: alpha beta gamma and x y z
        kind = 'psi4' if kind is None else kind
        domain = 'time' if domain is None else domain

        #
        gwylmo = this.gwylmo

        #
        tag = 'td' if domain in ('t','td','time') else 'fd'

        # Extract useful info
        x,y,z = this.radiation_axis[tag+'_x'],this.radiation_axis[tag+'_y'],this.radiation_axis[tag+'_z']
        alpha,beta,gamma = this.radiation_axis[tag+'_alpha'],this.radiation_axis[tag+'_beta'],this.radiation_axis[tag+'_gamma']
        domain_vals = this.radiation_axis[tag+'_domain']

        #
        fig = figure( figsize=4*figaspect(1) )
        clr = rgb(3,jet=True); grey = ones(3)*0.8
        lw = 1.5

        #
        domain_min = domain_vals[gwylmo.preinspiral.right_index] if domain in ('t','time') else gwylmo.wstart_pn/(2*pi)
        domain_max = domain_vals[gwylmo.postringdown.left_index] if domain in ('t','time') else gwylmo.lm[2,2][kind].dt/pi

        #
        mask = (domain_vals>=domain_min) & (domain_vals<=domain_max)

        #
        ax = subplot(3,1,1)
        title( gwylmo.simname )
        if domain in ('t','time'):
            plot( domain_vals, gwylmo.lm[2,2][kind].plus, color=grey, linewidth = lw )
            plot( domain_vals, gwylmo.lm[2,2][kind].cross, color=0.8*grey, linewidth = lw )
            plot( domain_vals, gwylmo.lm[2,2][kind].amp, linewidth = lw, label=r'$\psi_{22}$' )
            yscale('log',nonposy='clip')
        else:
            plot( domain_vals, gwylmo.lm[2,2][kind].fd_amp, linewidth = lw, label=r'$\psi_{22}$' )
            xscale('log'); yscale('log')
            ylim( lim(gwylmo.lm[2,2][kind].fd_amp[mask]) )
        grid()
        legend( frameon=False, loc='best' )

        #
        subplot(3,1,2,sharex=ax)
        reshift = lambda V: V - V[mask][0] + mod(V[mask][0],2*pi)
        plot( abs(domain_vals), reshift(alpha), color = clr[0], linewidth = lw, label=r'$\alpha$' )
        plot( abs(domain_vals), reshift(beta),  color = clr[1], linewidth = lw, label=r'$\beta$' )
        plot( abs(domain_vals), reshift(gamma), color = clr[2], linewidth = lw, label=r'$\gamma$' )
        legend( frameon=False, loc='best' )
        ylim( lim( hstack([reshift(alpha)[mask],reshift(beta)[mask],reshift(gamma)[mask]]), dilate=0.1 ) )
        grid()

        #
        subplot(3,1,3,sharex=ax)
        plot( abs(domain_vals), reflect_unwrap(x), color = clr[0], linewidth = lw, label=r'$x$' )
        plot( abs(domain_vals), y,  color = clr[2], linewidth = lw, label=r'$y$' )
        plot( abs(domain_vals), z, color = clr[1], linewidth = lw, label=r'$z$' )
        legend( frameon=False, loc='best' )
        ylim( lim( hstack([x[mask],y[mask],z[mask]]), dilate=0.1 ) )
        grid()
        xlabel( '$t/M$' if 'td'==tag else '$fM$' )

        #
        ax.set_xlim( [ domain_min, domain_max ] )

        #
        if this.save:
            filepath = join( this.outdir,'%s_%s_3panel.pdf'%(gwylmo.simname,tag))
            savefig(filepath,pad_inches=0, bbox_inches='tight')
            close_figure()
            # show()

    #
    def plot_radiation_axis_on_sphere( this, domain=None, kind = None, view = None ):

        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import figure, figaspect, plot, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, subplot, grid, title, draw, show, savefig, axis
        from matplotlib.pyplot import close as close_figure
        from numpy import mod,pi,hstack,array,ones,linalg,arange,zeros_like
        from os.path import join

        # Calc radiation axis: alpha beta gamma and x y z
        kind = 'psi4' if kind is None else kind
        domain = 'time' if domain is None else domain

        #
        gwylmo = this.gwylmo

        #
        if view is None:
            view = (30,-60)

        #
        tag = 'td' if domain in ('t','td','time') else 'fd'

        # Extract useful info
        x,y,z = this.radiation_axis[tag+'_x'],this.radiation_axis[tag+'_y'],this.radiation_axis[tag+'_z']
        alpha,beta,gamma = this.radiation_axis[tag+'_alpha'],this.radiation_axis[tag+'_beta'],this.radiation_axis[tag+'_gamma']
        domain_vals = this.radiation_axis[tag+'_domain']

        #
        fig = figure( figsize=4*figaspect(1) )
        ax = fig.add_subplot(111, projection='3d')
        color = rgb(3)

        #
        plot_3d_mesh_sphere( ax, color='k', alpha=0.05, lw=1 )

        #
        gwylmo.__calc_radiated_quantities__(use_mask=False)
        k = 0
        remnant = gwylmo.old_remnant if 'old_remnant' in gwylmo.__dict__ else gwylmo.remnant
        jx,jy,jz = remnant['J'][k] / linalg.norm( remnant['J'][k] )
        jfx,jfy,jfz = remnant['J'][-1] / linalg.norm( remnant['J'][-1] )

        #
        if tag == 'td':
            #
            mask = arange( gwylmo.startindex+1, gwylmo.endindex_by_frequency+1 )
        else:
            mask = (abs(gwylmo.f)>gwylmo.wstart_pn/(2*pi)) & (abs(gwylmo.f)<gwylmo.lm[2,2][kind].dt/4)


        #
        lx,ly,lz = (gwylmo.L1+gwylmo.L2)/linalg.norm( gwylmo.L1+gwylmo.L2 )#
        ax.scatter( lx,ly,lz, marker='h', color='lawngreen', label='Initial $L$ (nrutils)',edgecolors='k' )

        #
        ax.scatter( jx,jy,jz,marker='o', c='dodgerblue', label='Initial $J$ (Radiated Est.)' )
        ax.scatter( jfx,jfy,jfz,marker='s', c='dodgerblue', label='Final $J$ (Radiated Est.)',s=80,alpha=0.5 )

        #
        J = remnant['J']
        absJ = zeros_like(J)
        for k in range(J.shape[0]):
            J[k] /= linalg.norm(J[k])
            absJ[k] = J[k]
        plot( J[:,0],J[:,1],J[:,2], label='$J(t)$ (Radiated Est.)' )

        #
        S = gwylmo.S
        L = gwylmo.L
        bbh_jx,bbh_jy,bbh_jz = (L+S)/linalg.norm( L+S )
        ax.scatter( bbh_jx,bbh_jy,bbh_jz, label='Initial $J$ (BBH)', color='tomato', marker='o' )

        #
        sfx,sfy,sfz = gwylmo.Sf/linalg.norm(gwylmo.Sf)
        ax.scatter( sfx,sfy,sfz, color='tomato', label='Final $J$ (BBH)', marker='v' )

        #
        plot( x[mask],y[mask],z[mask], lw=2, color='grey', label='$\hat{V}$' )

        #
        xlabel('$x$')
        ylabel('$y$');

        axlim = 0.64*array([-1,1])
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
        ax.set_zlim(axlim)

        axis('off')

        #
        ax.view_init(view[0],view[1])

        #
        legend( loc=1, frameon=True )

        #
        if this.save:
            filepath = join( this.outdir,'%s_%s_sphere_el%i_az%i.pdf'%(gwylmo.simname,tag,view[0],view[1]))
            savefig(filepath,pad_inches=0, bbox_inches='tight')
            close_figure()
            # show()


# Calculate Widger D-Matrix Element
def wdelement( ll,         # polar index (eigenvalue) of multipole to be rotated (set of m's for single ll )
               mp,         # member of {all em for |em|<=l} -- potential projection spaceof m
               mm,         # member of {all em for |em|<=l} -- the starting space of m
               alpha,      # -.
               beta,       #  |- Euler angles for rotation
               gamma ):    # -'

    #** James Healy 6/18/2012
    #** wignerDelement
    #*  calculates an element of the wignerD matrix
    # Modified by llondon6 in 2012 and 2014
    # Converted to python by spxll 2016
    #
    # This implementation apparently uses the formula given in:
    # https://en.wikipedia.org/wiki/Wigner_D-matrix
    #
    # Specifically, this the formula located here: https://wikimedia.org/api/rest_v1/media/math/render/svg/53fd7befce1972763f7f53f5bcf4dd158c324b55

    #
    from numpy import sqrt,exp,cos,sin,ndarray
    from scipy.misc import factorial

    #
    if ( (type(alpha) is ndarray) and (type(beta) is ndarray) and (type(gamma) is ndarray) ):
        alpha,beta,gamma = alpha.astype(float), beta.astype(float), gamma.astype(float)
    else:
        alpha,beta,gamma = float(alpha),float(beta),float(gamma)

    #
    coefficient = sqrt( factorial(ll+mp)*factorial(ll-mp)*factorial(ll+mm)*factorial(ll-mm))*exp( 1j*(mp*alpha+mm*gamma) )

    # NOTE that there may be convention differences where the overall sign of the complex exponential may be negated

    #
    total = 0

    # find smin
    if (mm-mp) >= 0 :
        smin = mm - mp
    else:
        smin = 0

    # find smax
    if (ll+mm) > (ll-mp) :
        smax = ll-mp
    else:
        smax = ll+mm

    #
    if smin <= smax:
        for ss in range(smin,smax+1):
            A = (-1)**(mp-mm+ss)
            A *= cos(beta/2)**(2*ll+mm-mp-2*ss)  *  sin(beta/2)**(mp-mm+2*ss)
            B = factorial(ll+mm-ss) * factorial(ss) * factorial(mp-mm+ss) * factorial(ll-mp-ss)
            total += A/B

    #
    element = coefficient*total
    return element

# Calculate Widner D Matrix
def wdmatrix( l,                # polar l
              mrange,    # range of m values
              alpha,
              beta,
              gamma,
              verbose = None ): # let the people know

    #
    from numpy import arange,array,zeros,complex256

    # Handle the mrange input
    if mrange is None:
        #
        mrange = arange( -l, l+1 )
    else:
        # basic validation
        for m in mrange:
            if abs(m)>l:
                msg = 'values in m range must not be greater than l'
                error(msg,'wdmatrix')

    #
    dim = len(mrange)
    D = zeros( (dim,dim), dtype=complex256 )
    for j,mm in enumerate(mrange):
        for k,mp in enumerate(mrange):
            D[j,k] = wdelement( l, mp, mm, alpha, beta, gamma )

    #
    return D

# # Given an array of complex valued waveform timeseries, and the related mutipolar spherical indeces, as well as the desired rotation angles, rotate the waveform set
# def mprotate( mpdict,           # dictionary or (l,m): complex__time_series_multipole
#               angles,           # three euler angles in order alpha beta gamma
#               verbose = None ): # Let the people know
#
#     # Import useful things
#     from numpy import array, arange, dot
#
#     # Validate the mpdict input
#
#     # Build list of l values; roations will be allied one l at a time
#     lrange = sorted( list(set( [ lm[0] for lm in mpdict.keys() ] )) )
#
#     # A basic check for angles input, more needed
#     if len(angles) != 3:
#         msg = 'angles input must be three floats in alpha beta gamma order'
#         error(msg,'mprotate')
#     for ang in angles:
#         if not isinstance(ang,(float,int)):
#             msg = 'angles must be float or int'
#             error(msg,'mprotate')
#
#     # For each l value
#     for l in lrange:
#
#         # Calculate the m range to use
#         mrange = sorted( [ lm[-1] for lm in mpdict.keys() if l==lm[0] ] )
#
#         # Calculate the related d matrix
#         alpha,beta,gamma = angles
#         D = wdmatrix( l, mrange, alpha, beta, gamma )
#
#         # For all time coordinates (the related indeces)
#         tindmap = range( len( mpdict[ mpdict ] ) )
