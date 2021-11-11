
#
from nrutils.core.basics import *
from positive.maths import wdelement,wdmatrix

#
class gwylm_radiation_axis_workflow:

    '''
    Workflow to calculate optimal emission axis (i.e. radiation axis), and related quantities. Comprehensive plotting is suggested and included as an option.
    '''

    #
    def __init__( this, gwylmo, kind=None, plot=False, outdir=None, save=False, safe_domain_range=None, verbose=True,__format__=None, ref_orientation=None, domain='both', select_lm_list=None ):

        #
        from os.path import expanduser
        from numpy import isnan,array

        # Calculate radiated and remnant quantities
        alert('Calculating Radiated Quantities','gwylm_radiation_axis_workflow',verbose=verbose)
        gwylmo.__calc_radiated_quantities__(use_mask=False)

        #
        allowed_domains = ('time', 'td', 'freq', 'fd', 'both')
        if not( domain in allowed_domains ):
            error('domain keyword value must be in '+str(allowed_domains)+' but '+red(bold(str(domain)))+' found.')

        # Store reference to input gwylmo
        this.gwylmo = gwylmo
        
        # Store a list of multipole moments to use for calculation of angles; if None then all multipole moments are used
        this.select_lm_list = select_lm_list

        # warn if not in J related frame
        if not ( 'j' in str(gwylmo.frame).lower() ):
            warning('This function works best if the gwylm object input is in a J-aligned frame, but %s frame found' %
                    bold(red(gwylmo.frame)))

        # NOTE that relevant information will be stored within the current object
        this.radiation_axis = {}
        rax = this.radiation_axis

        #
        if kind is None: 
            kind = 'psi4'
            warning('No "kind" keyword input. We will proceed using kind='+red('psi4')+' to calculate the system\'s radiation axis.'+bold(red(' To turn off this warning, manually set the kind keyword to "psi4", "strain" or "news".')) )

        #
        if not ( kind in ('psi4','strain','news') ):
            error('The kind keyword input must be in ("psi4","strain","news"), but '+red(str(kind))+' found.')

        #
        this.save = save

        #
        if outdir is None: outdir = '~/Desktop/'+gwylmo.simname
        if save:
            outdir = expanduser( outdir ); this.outdir = outdir
            mkdir( this.outdir, verbose=verbose )
            
        #
        this.process_td = domain.lower() in ('time', 'both', 't', 'td')
        this.process_fd = domain.lower() in ('freq', 'both', 'f', 'fd')

        # Calculate radiation axes in time and frequency domain
        if this.process_td:
            
            alert('Calculating TD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
            td_alpha,td_beta,td_gamma,td_x,td_y,td_z,td_domain = this.calc_radiation_axis( domain = 'time', kind = kind, safe_domain_range=None, __format__=__format__  )
            
            # Store time domain data
            rax['td_alpha'], rax['td_beta'], rax['td_gamma'] = td_alpha, td_beta, td_gamma
            rax['td_x'], rax['td_y'], rax['td_z'] = td_x, td_y, td_z
            rax['td_domain'] = td_domain

        if this.process_fd:
            
            alert('Calculating FD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
            fd_alpha,fd_beta,fd_gamma,fd_x,fd_y,fd_z,fd_domain = this.calc_radiation_axis( domain = 'freq', kind = kind, safe_domain_range=safe_domain_range, __format__=__format__  )
            
            # Store freq domain data
            rax['fd_alpha'], rax['fd_beta'], rax['fd_gamma'] = fd_alpha, fd_beta, fd_gamma
            rax['fd_x'], rax['fd_y'], rax['fd_z'] = fd_x, fd_y, fd_z
            rax['fd_domain'] = fd_domain

        # Define plotting function to be called either internally or externally
        def internal_plotting_function():

            #
            if this.process_td:
                alert('Plotting TD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
                this.plot_radiation_axis_3panel( domain='time' )
            
            if this.process_fd:
                alert('Plotting FD Radiation Axis Series','gwylm_radiation_axis_workflow',verbose=verbose)
                this.plot_radiation_axis_3panel( domain='freq' )

            #
            view = None
            if this.process_td: this.plot_radiation_axis_on_sphere( domain='time', view=view )
            if this.process_fd: this.plot_radiation_axis_on_sphere( domain='freq', view=view )
            #
            view = (90,270)
            if this.process_td: this.plot_radiation_axis_on_sphere(domain='time', view=view)
            if this.process_fd: this.plot_radiation_axis_on_sphere( domain='freq', view=view )

        # Store internal plotting funtion to this object
        this.plot = internal_plotting_function

        #
        if plot: this.plot()

    # Encapsulation of calc angles given domain and type
    def calc_radiation_axis( this, domain=None, kind = None, safe_domain_range=None, __format__=None, ref_orientation=None,select_lm_list=None  ):

        # Import usefuls
        from numpy import pi

        # Calc radiation axis: alpha beta gamma and x y z
        kind = 'psi4' if kind is None else kind
        domain = 'time' if domain is None else domain

        #
        gwylmo = this.gwylmo.copy()
        
        #
        if select_lm_list is None:
            select_lm_list = this.select_lm_list
        
        #
        if select_lm_list is not None:
            #
            gwylmo = gwylmo.selectlm( select_lm_list )

        # Construct dictionary of multipoles using all multipoles available
        # NOTE that the __format__ toggle may be useful for sanity checks
        if not ( __format__ is None ):
            alert('Using alternative dictionary format')
            mp = { (l,m) : ( {'real':gwylmo.lm[l,m][kind].wfarr[:,1],'imag':gwylmo.lm[l,m][kind].wfarr[:,2]} if domain in ('t','time') else {'real':gwylmo.lm[l,m][kind].fd_wfarr[:,1],'imag':gwylmo.lm[l,m][kind].fd_wfarr[:,2]} ) for l,m in gwylmo.lm  }
        else:
            mp = { (l,m) : ( gwylmo.lm[l,m][kind].y if domain in ('t','time') else gwylmo.lm[l,m][kind].fd_y ) for l,m in gwylmo.lm  }
        # Domain values: time or freq
        domain_vals = gwylmo.lm[2,2][kind].t if domain in ('t','time') else gwylmo.lm[2,2][kind].f

        #
        if safe_domain_range is None:
            domain_min = domain_vals[gwylmo.preinspiral.right_index] if domain in ('t','time') else gwylmo.wstart_pn/(2*pi)
            domain_max = domain_vals[gwylmo.postringdown.left_index] if domain in ('t','time') else gwylmo.lm[2,2][kind].dt/pi
            safe_domain_range = [domain_min,domain_max]

        # Calculate corotating angles using low-level function
        alpha,beta,gamma,x,y,z = calc_coprecessing_angles( mp, domain_vals, ref_orientation=gwylmo.L if ref_orientation is None else ref_orientation, return_xyz='all', safe_domain_range = safe_domain_range )
        #
        # #
        # from matplotlib.pyplot import plot,title,subplot,show,figure,xlim
        # figure()
        # subplot(1,3,1)
        # plot( domain_vals, alpha )
        # if domain in ('f','freq') : xlim([0.01,0.1])
        # subplot(1,3,2)
        # plot( domain_vals, beta )
        # if domain in ('f','freq') : xlim([0.01,0.1])
        # subplot(1,3,3)
        # plot( domain_vals, gamma )
        # if domain in ('f','freq') : xlim([0.01,0.1])
        # title('hi!!!')
        # show()

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
        ax1 = subplot(3,1,1)
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
        ax2 = subplot(3,1,2,sharex=ax1)
        reshift = lambda V: V - V[mask][0] + mod(V[mask][0],2*pi)
        plot( abs(domain_vals), reshift(alpha), color = clr[0], linewidth = lw, label=r'$\alpha$' )
        plot( abs(domain_vals), reshift(beta),  color = clr[1], linewidth = lw, label=r'$\beta$' )
        plot( abs(domain_vals), reshift(gamma), color = clr[2], linewidth = lw, label=r'$\gamma$' )
        legend( frameon=False, loc='best' )
        ylim( lim( hstack([reshift(alpha)[mask],reshift(beta)[mask],reshift(gamma)[mask]]), dilate=0.1 ) )
        grid()

        #
        ax3 = subplot(3,1,3,sharex=ax1)
        plot( abs(domain_vals), reflect_unwrap(x), color = clr[0], linewidth = lw, label=r'$x$' )
        plot( abs(domain_vals), y,  color = clr[2], linewidth = lw, label=r'$y$' )
        plot( abs(domain_vals), z, color = clr[1], linewidth = lw, label=r'$z$' )
        legend( frameon=False, loc='best' )
        ylim( lim( hstack([x[mask],y[mask],z[mask]]), dilate=0.1 ) )
        grid()
        xlabel( '$t/M$' if 'td'==tag else '$fM$' )

        #
        ax1.set_xlim( [ domain_min, domain_max ] )

        #
        if this.save:
            filepath = join( this.outdir,'%s_%s_3panel.pdf'%(gwylmo.simname,tag))
            savefig(filepath,pad_inches=0, bbox_inches='tight')
            close_figure()
            # show()

        #
        return (fig,(ax1,ax2,ax3))

    #
    def plot_radiation_axis_on_sphere( this, domain=None, kind = None, view = None, ax = None, v_color='grey' ):

        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import figure, figaspect, plot, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, subplot, grid, title, draw, show, savefig, axis, gcf, gca
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
        if ax is None:
            fig = figure( figsize=4*figaspect(4) )
            # fig = figure()
            #ax = fig.add_subplot(111, projection='3d')
            ax = gca(projection='3d')
            
            # See: https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-634964954
            ax.set_box_aspect((1, 1, 1))
            
        else:
            fig = gcf()
        
        #
        ax.view_init(view[0],view[1])#
        plot_3d_mesh_sphere( ax, color='k', alpha=0.05, lw=1 )

        #
        color = rgb(3)



        #
        gwylmo.__calc_radiated_quantities__(use_mask=False)
        k = 0
        # remnant = gwylmo.old_remnant if 'old_remnant' in gwylmo.__dict__ else gwylmo.remnant
        remnant = gwylmo.remnant
        jx,jy,jz = remnant['J'][k] / linalg.norm( remnant['J'][k] )
        jfx,jfy,jfz = remnant['J'][-1] / linalg.norm( remnant['J'][-1] )

        #
        if tag == 'td':
            #
            mask = arange( gwylmo.startindex+1, gwylmo.endindex_by_frequency+1 )
        else:
            mask = (abs(gwylmo.f)>gwylmo.wstart_pn/(2*pi)) & (abs(gwylmo.f)<gwylmo.lm[2,2][kind].dt/4)


        #
        lx,ly,lz = (gwylmo.L1+gwylmo.L2)/linalg.norm( gwylmo.L1+gwylmo.L2 )
        ax.scatter( lx,ly,lz, marker='s', color='lawngreen', label='Initial $L$ (bbh-File)',edgecolors='k',zorder=20 )

        #
        ax.scatter( jx,jy,jz,marker='o', c='dodgerblue', label='Initial $J$ (Radiated Est.)' )
        ax.scatter( jfx,jfy,jfz,marker='s', c='dodgerblue', label='Final $J$ (Radiated Est.)',s=80,alpha=0.5 )

        #
        # J = this.gwylmo.remnant['J'].copy()
        J = remnant['J']
        # J /= linalg.norm(J,axis=1)
        for k in range(J.shape[0]):
            J[k] /= linalg.norm(J[k])
        plot( J[:,0],J[:,1],J[:,2], label='$J(t)$ (Radiated Est.)',zorder=-2e4 )

        #
        S = gwylmo.S
        L = gwylmo.L
        bbh_jx,bbh_jy,bbh_jz = (L+S)/linalg.norm( L+S )
        ax.scatter( bbh_jx,bbh_jy,bbh_jz, label='Initial $J$ (bbh-File)', color='tomato', marker='o' )

        #
        sfx,sfy,sfz = gwylmo.Sf/linalg.norm(gwylmo.Sf)
        ax.scatter( sfx,sfy,sfz, color='tomato', label='Final $J$ (bbh-File)', marker='v' )

        #
        plot( x[mask],y[mask],z[mask], lw=2, color=v_color, label='$\hat{V}(%s)$ (Radiated Est.)'%('t' if tag=='td' else 'f') )

        #
        xlabel('$x$')
        ylabel('$y$');

        axlim = 0.64*array([-1,1])
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
        ax.set_zlim(axlim)

        axis('off')

        this.gwylmo.plot_3d_L(ax=ax)

        #
        legend( loc=3, frameon=True )
        title(('Time Domain\n' if tag=='td' else 'Frequency Domain\n' ) + this.gwylmo.simname, loc='left', alpha=0.5, fontsize=12)

        #
        if this.save:
            filepath = join( this.outdir,'%s_%s_sphere_el%i_az%i.pdf'%(gwylmo.simname,tag,view[0],view[1]))
            savefig(filepath,pad_inches=0, bbox_inches='tight')
            close_figure()
            # show()

        #
        return fig,ax



    
#
def calc_ulterior_angles_helper( lmlist, A,B, ALPHA, BETA, GAMMA, MU=1,smalld_splines=None ):

    '''

    Given 

    lmlist -- List (l,m) in same order in m as A and B inputs. l is fixed
    A -- Inertial momements at a single frequency 
    B -- Coprecessing moments at a single fequency (Fourier transform of TD coprecessing moments)

    ALPHA, BETA, GAMMA -- Euler angles for 3D rotation. Each is a dict containing eg alpha_lm with keys (l,m). These dictionaries encode 3*(2*l+1) angles :/

    Determine

    C -- The rotation of B by (-gamma,-beta,-alpha)
    MU -- a distance measurement between A and C

    Return

    MU -- single number 

    '''
    
    #
    from numpy import array
    from nrutils import rotate_wfarrs_at_all_times

    #
    C = []
    for k,(l,m) in enumerate(lmlist):

        #
        Cm = MU * rotate_wfarrs_at_all_times( l,m, { (ll,mp):B[k] for k,(ll,mp) in enumerate(lmlist) }, (GAMMA,BETA,ALPHA),angles_are_ulterior=True,smalld_splines=smalld_splines )
        C.append( Cm )
    #
    C = array(C)

    # Given C, estiamte the square "distance" between A and C
    Z = sum(abs(  (A[:,1]+1j*A[:,2])  -  (C[:,1]+1j*C[:,2])  )**2)

    #
    return Z


#
def calc_ulterior_angles( preimage_gwylmo, image_gwylmo, l, kind='psi4', flim=None ):
    '''
    
    INPUTS
    ---
    
    '''
    
    #
    from numpy import array,pi,zeros_like,mod
    from scipy.fftpack import fft, fftfreq, fftshift, ifft
    from positive.maths import wdelement,wigner_smalld_splines
    from scipy.optimize import minimize,basinhopping,brute
    
    #
    lmlist = [ (ll,mm) for ll,mm in preimage_gwylmo.__lmlist__ if ll==l]
    FT = lambda X: fftshift( fft( X ) )
    
    #
    f_domain = image_gwylmo.f
    mask = range(len(f_domain))
    
    #
    if flim:
        mask = (f_domain>min(flim)) & (f_domain<max(flim))
        f_domain = f_domain[mask]
    
    #
    NumM = (2*l+1)
    LENX = 3*NumM
    X   = []
    ERR = []
    
    # Generate splines to spped up Wigner D-Matrix evaluation
    smalld_splines = wigner_smalld_splines(l)
    
    #
    for k,f in enumerate(f_domain):
    
    
        '''
        * We will consider the k'th frequency bin
        * Let Alm be the inertial frame multipole moments at a single frequency bin
        * Let Blm be the coprecessing frame moments at a single frequency bin 
        '''
        A = array([ preimage_gwylmo[l,m][kind].fd_wfarr[mask][k,:] for l,m in lmlist ])
        B = array([    image_gwylmo[l,m][kind].fd_wfarr[mask][k,:] for l,m in lmlist ])
        
        #
        def action(X):
            
            #
            lenX = len(X)
            if lenX != (LENX+1):
                error('input must have length of 3*(2*l+1)+1=%i'%(LENX+1))
            if mod(lenX-1,3):
                error('input must be have len-1 of integer multiple of 3')
                
            # Create lists by unpacking vectorized input 
            alpha_list  = X[        0 : NumM     ]
            beta_list   = X[     NumM : (2*NumM) ]
            gamma_list  = X[ (2*NumM) : (3*NumM) ]
            mu_value    = X[3*NumM]
            
            # Create dicts by assuming m ordering in lmlist
            ALPHA = { mp:alpha_list[j] for j,(_,mp) in enumerate(lmlist) }
            BETA  = { mp:beta_list[ j]  for j,(_,mp) in enumerate(lmlist) }
            GAMMA = { mp:gamma_list[j] for j,(_,mp) in enumerate(lmlist) }
            MU    = mu_value
            
            # Calculate and return distance estimate
            return calc_ulterior_angles_helper( lmlist, A,B, ALPHA, BETA, GAMMA, MU,smalld_splines=smalld_splines )
        
        #
        initial_guess = zeros_like(range(0,LENX+1))
        if k>0: initial_guess = X[-1]
        if k==0:
            foo = basinhopping( action, initial_guess  )
        else:
        # bounds = [(0,2*pi) for k in initial_guess]
            foo = minimize( action, initial_guess  )
        x0 = foo.x 
        fval = foo.fun
        
        X.append(x0)
        ERR.append(fval)
        if mod(k,20)==0:
            print('.%1.2f(%e)..'%(100*float(k)/len(f_domain),foo.fun),end='')
        
    #
    return (X,ERR,f_domain)
        
            
    

# # Calculate Widger D-Matrix Element
# def wdelement( ll,         # polar index (eigenvalue) of multipole to be rotated (set of m's for single ll )
#                mp,         # member of {all em for |em|<=l} -- potential projection spaceof m
#                mm,         # member of {all em for |em|<=l} -- the starting space of m
#                alpha,      # -.
#                beta,       #  |- Euler angles for rotation
#                gamma ):    # -'

#     #** James Healy 6/18/2012
#     #** wignerDelement
#     #*  calculates an element of the wignerD matrix
#     # Modified by llondon6 in 2012 and 2014
#     # Converted to python by spxll 2016
#     #
#     # This implementation apparently uses the formula given in:
#     # https://en.wikipedia.org/wiki/Wigner_D-matrix
#     #
#     # Specifically, this the formula located here: https://wikimedia.org/api/rest_v1/media/math/render/svg/53fd7befce1972763f7f53f5bcf4dd158c324b55

#     #
#     from numpy import sqrt,exp,cos,sin,ndarray# Reference factorial from scipy
#     try:
#         from scipy.misc import factorial
#     except:
#         from scipy.special import factorial

#     #
#     if ( (type(alpha) is ndarray) and (type(beta) is ndarray) and (type(gamma) is ndarray) ):
#         alpha,beta,gamma = alpha.astype(float), beta.astype(float), gamma.astype(float)
#     else:
#         alpha,beta,gamma = float(alpha),float(beta),float(gamma)

#     #
#     coefficient = sqrt( factorial(ll+mp)*factorial(ll-mp)*factorial(ll+mm)*factorial(ll-mm))*exp( 1j*(mp*alpha+mm*gamma) )

#     # NOTE that there may be convention differences where the overall sign of the complex exponential may be negated

#     #
#     total = 0

#     # find smin
#     if (mm-mp) >= 0 :
#         smin = mm - mp
#     else:
#         smin = 0

#     # find smax
#     if (ll+mm) > (ll-mp) :
#         smax = ll-mp
#     else:
#         smax = ll+mm

#     #
#     if smin <= smax:
#         for ss in range(smin,smax+1):
#             A = (-1)**(mp-mm+ss)
#             A *= cos(beta/2)**(2*ll+mm-mp-2*ss)  *  sin(beta/2)**(mp-mm+2*ss)
#             B = factorial(ll+mm-ss) * factorial(ss) * factorial(mp-mm+ss) * factorial(ll-mp-ss)
#             total += A/B

#     #
#     element = coefficient*total
#     return element

# # Calculate Widner D Matrix
# def wdmatrix( l,                # polar l
#               mrange,    # range of m values
#               alpha,
#               beta,
#               gamma,
#               verbose = None ): # let the people know

#     #
#     from numpy import arange,array,zeros,complex256

#     # Handle the mrange input
#     if mrange is None:
#         #
#         mrange = arange( -l, l+1 )
#     else:
#         # basic validation
#         for m in mrange:
#             if abs(m)>l:
#                 msg = 'values in m range must not be greater than l'
#                 error(msg,'wdmatrix')

#     #
#     dim = len(mrange)
#     D = zeros( (dim,dim), dtype=complex256 )
#     for j,mm in enumerate(mrange):
#         for k,mp in enumerate(mrange):
#             D[j,k] = wdelement( l, mp, mm, alpha, beta, gamma )

#     #
#     return D
