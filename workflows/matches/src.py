
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Import useful things
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
import os, pickle
os.system('clear')
from nrutils import scsearch,gwylm,physf,physhf,eta2q,lalphenom
from nrutils.analyze.match import match as match_object
from positive import *
import lalsimulation as lalsim
from numpy import pi,array,savetxt,log, angle, unwrap, average, linspace, arange, diff, ones, ones_like, zeros, zeros_like, hstack, ndarray
from shutil import copyfile
from glob import glob
executable_name = glob( parent( os.path.realpath(__file__) )+'*work' )[0].split('/')[-1]
from scipy.interpolate import InterpolatedUnivariateSpline as spline
# Setup plotting backend
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 0.8
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 26
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.formatter.useoffset'] = True # False
# Plotting functions
from matplotlib.pyplot import tight_layout, gcf, sca, plot, xlabel, ylabel, title, legend, xscale, yscale, xlim, ylim, subplot, figure, savefig, axvline, axhline, fill_between, xticks, yticks, gca, close

alert("We are getting lalsimulation from: "+yellow(bold(lalsim.__path__[0])),"work",heading=True)

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Load content of config.ini
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
thisdir = parent( os.path.realpath(__file__) )
inipath = osjoin( thisdir, 'config.ini' )
defaults = { 'M_Sol':100, 'fmin':'', 'fmax':400, 'diagnostic_inclination':'pi/2', 'diagnostic_phase':'pi/2', 'D_Mpc':450, 'outdir':'', 'N_inclinations':21, 'N_PSI_signal':8, 'N_phi_signal':11  }
config = smart_object( inipath, cleanup=True, comment=['#',';'], defaults=defaults )


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Handle Configuration Options
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
# Convert nrlmlist to list if needed
if len(config.nrlmlist)>0: config.nrlmlist = eval( ','.join( config.nrlmlist ) )
# Handle diagnostic values
config.diagnostic_inclination = eval(str(config.diagnostic_inclination).lower()) if len(str(config.diagnostic_inclination))>0 else pi/2
config.diagnostic_phase = eval(str(config.diagnostic_phase).lower()) if len(str(config.diagnostic_phase))>0 else pi/2
# Handle the output directory
config.outdir = os.path.expanduser(config.outdir) if len(config.outdir)>0 else thisdir
# make sure the sim keyword is list
if isinstance(config.simulation_keywords,str): config.simulation_keywords = [config.simulation_keywords]

#
for simkey in config.simulation_keywords:

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Find the NR Simulation
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    alert('Finding simulation with keyword %s'%green(simkey),heading=True)
    sce = scsearch( keyword=simkey, verbose=True )
    if len(sce)>1:
        error('More than one simulation has been found for this keyowrd. Please refine your jeyword so that this is not the case.')
    else:
        sce = sce[-1]


    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Create directories for I/O
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    alert('Create directories for I/O',heading=True)
    # Base directory for output
    basedir = osjoin(config.outdir,sce.simname+'/')
    # Cofiguration and source files: mkdir and copy files
    config_dir = basedir
    mkdir(config_dir,verbose=True)
    copyfile( os.path.realpath(__file__) , osjoin(config_dir,os.path.basename(__file__)) )
    copyfile(inipath, osjoin(config_dir,inipath.split('/')[-1]) )
    copyfile(osjoin(thisdir,executable_name),osjoin(config_dir,executable_name) )
    # Test plots
    diagnostics_dir = osjoin( basedir, 'diagnostics/' )
    mkdir(diagnostics_dir,verbose=True)
    # ascii data
    data_dir = osjoin( basedir, 'data/' )
    mkdir(data_dir,verbose=True)


    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Load the simulation data
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    alert('Load NR data %s. Remove junk radiation. Calculate strain (FFI).'%(cyan('(%s)'%('all multipoles' if len(config.nrlmlist)==0 else 'select multipoles'))),heading=True)
    for k in list(config.nrlmlist):
        config.nrlmlist.append(k)
        config.nrlmlist.append( (k[0],-k[-1]) )
    config.nrlmlist = list(set(config.nrlmlist))
    if len(config.nrlmlist) > 0:
        y = gwylm(sce,lm=config.nrlmlist,clean=True,verbose=True,dt=0.5,pad=500)
    else:
        y = gwylm(sce,lmax=5,clean=True,verbose=True,dt=0.5,pad=500)

    # Let the people know what the settings are
    alert('The match workflow will be performed with the following settings:',heading=True)
    config.show()

    # -- Define shorthand identifiers -- #
    theta = config.diagnostic_inclination
    phi = config.diagnostic_phase
    M_Sol = config.M_Sol
    # Determine the Total mass from the mass ratio and M2
    D_Mpc = config.D_Mpc
    df = physf(y.ylm[0].df,M_Sol)
    # Make sure that the defulat behavior of fmin is handled
    nrfmin = physf( y.wstart_pn/(2*pi), M_Sol ) + 10.0
    if config.fmin < nrfmin: warning('The input fmin (%f) is less than the expect fmin of the NR data (%f)'%(config.fmin,nrfmin))
    config.fmin = eval(str(config.fmin)) if len(str(config.fmin))>0 else nrfmin
    fmin = config.fmin
    fmax = config.fmax

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Make and save a diagnostic plot of the NR waveform in the TD
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    alert('Make and save a diagnostic plot of the NR waveform in the TD',heading=True)
    degrees = [0,30,90,90+30,90+90]
    for deg in degrees:
        print '>> theta = %f (degrees)' % deg
        s = y.recompose( (pi*deg)/180, config.diagnostic_phase, select_lm=None,kind='strain',output_array=not True,domain='freq')
        figure( figsize = 2*array([6,5]) )
        axs,fig=s.plot(domain='time',fig=gcf())
        tight_layout(pad=2,h_pad=0.001,w_pad=1.2)
        sca(axs[0])
        title(r'$(\theta,\phi) = (%1.2f,%1.2f)~\text{(rad)}$'%((pi*deg)/180, config.diagnostic_phase))
        savefig( osjoin(diagnostics_dir,'nr_td_waveform_theta_%ideg.pdf'%deg), pad_inches=1 )
        close(gcf())
    print '>> Done.'


    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Make diagnostic plots of the waveform at a fixed inclination
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #

    #
    alert('Making diagnostic plots of ...',heading=True)

    # Define high-level waveform functions
    # -- For NR
    def signal_wfarr_fun( THETA,PHI,LM=None) :
        ans = physhf( y.recompose( THETA,PHI,kind='strain',domain='freq').fd_wfarr, M_Sol, D_Mpc )
        return ans
    # -- For MODEL
    def template_wfarr_fun( THETA,PHI,LM=None) :
        if LM is None: LM = []
        if set(LM) == set([(2,2),(2,-2)]):
            ans = lalphenom( y.eta,M_Sol,y.X1[-1],y.X2[-1],THETA,PHI,D_Mpc,df,fmin,fmax, approx='IMRPhenomD' )
        else:
            ans = lalphenom( y.eta,M_Sol,y.X1[-1],y.X2[-1],THETA,PHI,D_Mpc,df,fmin,fmax )
        return ans

    # Construct model and NR waveform at a fixed inclination and orbital phase
    phys_template = template_wfarr_fun( theta, phi )
    phys_signal = signal_wfarr_fun( theta,phi-pi/2 )
    # preformat for plotting
    mask = abs(phys_signal[:,0]) <= fmax
    phys_signal = phys_signal[mask,:]
    mask = abs(phys_template[:,0]) <= fmax
    phys_template = phys_template[mask,:]

    # Create match object
    mo = match_object( phys_template, phys_signal, fmin=fmin, fmax=fmax, signal_polarization=0, positive_f=True, psd_thing = lalsim.SimNoisePSDaLIGOZeroDetHighPower )

    # Make diagnistic plot of waveforms on noise curve
    print ' * '+yellow('Amplitudes and SNR on noise curve')
    fig = figure( figsize=3*array([4,4]) )
    mo.plot(fig=fig)
    title(r'$\theta=%1.2f$, no optimization'%theta)
    savefig( osjoin(diagnostics_dir,'compare_snr_and_amp_on_psd.pdf'), pad_inches=1 )
    close(gcf())

    # Plot phase derivative of NR and model
    print ' * '+yellow('FD phase derivatives')
    figure()
    f = phys_template[:,0].real
    d1 = spline_diff(f,unwrap(angle(phys_template[:,1])))
    d2 = spline_diff(f,unwrap(angle(phys_signal[:,1] )))
    k = (abs(f) <= config.fmax) & (abs(f) > config.fmin)
    plot( f, d2 - average(d2[k],weights=abs(phys_signal[k,1])), label='NR Data' )
    plot( f, d1 - average(d1[k],weights=abs(phys_template[k,1])), label='PhenomHM' )
    legend(frameon=False)
    ylim([-1,1])
    xlim([fmin/2,fmax])
    xscale('log')
    title('FD Phase Derivative')
    savefig( osjoin(diagnostics_dir,'compare_phase_derivatives.pdf'), pad_inches=1 )
    close(gcf())

    # Amplitudes
    print ' * '+yellow('FD amplitudes')
    figure()
    k = 1
    plot( abs(phys_signal[:,0]), abs(phys_signal[:,k]), label='NR Data' )
    plot( abs(phys_template[:,0]), abs(phys_template[:,k]), label='PhenomHM' )
    # ylim([1e-4,1e2])
    xlim([10,2e3])
    legend(frameon=False)
    yscale('log'); xscale('log')
    xlabel(r'$|f|$ (Hz)')
    ylabel(r'$|\tilde{h}_+(f)|$')
    title('FD Amplitudes')
    savefig( osjoin(diagnostics_dir,'compare_fd_amplitudes.pdf'), pad_inches=1 )
    close(gcf())

    # Template conjugate symmetry
    print ' * '+yellow('Conjugate symmetry')
    figure( figsize = 4*array([4.2,2] ) )
    j = phys_signal[:,0]>0
    k = phys_signal[:,0]<0
    subplot(1,2,1)
    title('signal conjugate symmetry?')
    plot( phys_signal[j,0].real, abs(phys_signal[j,1]) )
    plot(-phys_signal[k,0].real, abs(phys_signal[k,1].conj()), '-' )
    xscale('log')
    yscale('log')
    j = phys_template[:,0]>0
    k = phys_template[:,0]<0
    subplot(1,2,2)
    title('template conjugate symmetry?')
    plot( phys_template[j,0].real, abs(phys_template[j,1]) )
    plot( -phys_template[k,0].real, abs(phys_template[k,1].conj()), '-' )
    xscale('log')
    yscale('log')
    savefig( osjoin(diagnostics_dir,'test_conjugate_symmetry.pdf'), pad_inches=1 )
    close(gcf())

    # Double sided amplutdes
    print ' * '+yellow('Double sided amplutdes')
    figure( figsize = 6*array([4.2,2] ) )
    k = 1 # toggle for plus or cross
    plot( phys_signal[:,0].real, abs(phys_signal[:,k]), label='NR Data' )
    plot( phys_template[:,0].real, abs(phys_template[:,k]), label='PhenomHM'  )
    xlim(max(phys_signal[:,0]).real*array([-1,1]).real)
    yscale('log')
    xlabel(r'$f$ (Hz)')
    ylabel(r'$|\tilde{h}_+(f)|$')
    legend(frameon=False)
    savefig( osjoin(diagnostics_dir,'compare_double_sided_amplitudes.pdf'), pad_inches=1 )
    close(gcf())


    # Inspect effect of template polarization and verify that all methods for optizing over this agree
    print ' * '+yellow('Effect of template polarization on match (Method comarison)')
    psirange = linspace(0,pi,53)
    matchlist = []
    vanilla = []
    for psi in psirange:
        matchlist.append( mo.calc_basic_match(template_polarization=psi) )
        vanilla.append( mo.calc_template_pol_optimized_match() )
    matchlist = array(matchlist)
    vanilla = array(vanilla)
    figure()
    plot( psirange, matchlist, '-', label='No Optimization' )
    plot( psirange, vanilla, '--', label='Analytic Optimization' )
    xlabel(r'$\psi$ template')
    ylabel('match')
    title('Effect of Template Polarization')
    legend( frameon=not False, loc=6 )
    analytic_value =  mo.calc_template_pol_optimized_match()
    codomain = array( list(matchlist)+[analytic_value] )
    dy = diff(lim( codomain ))*0.1
    xlim( lim(psirange) )
    savefig( osjoin(diagnostics_dir,'compare_template_polarization_opt.pdf'), pad_inches=1 )
    close(gcf())


    # Effect of template orbital phase
    print ' * '+yellow('Effect of template orbital phase on match')
    figure()
    phi_range = linspace(0,2*pi,61)
    template_phi_wfarr_fun = lambda PHI: template_wfarr_fun( theta, PHI )
    matchlist = []
    for phi in phi_range:
        mo.apply( template_wfarr = template_phi_wfarr_fun(phi) )
        matchlist.append(  mo.calc_template_pol_optimized_match() )
    plot( phi_range, matchlist, color=0.5*ones(3), marker='o', mfc='none' )
    phi_ = linspace(0,2*pi,2e2)
    plot( phi_, spline(phi_range,matchlist)(phi_), '--k', alpha=0.4 )
    xlabel(r'$\phi_{\mathrm{template}}$ (Template Orbital Phase)')
    xlim(lim(phi_))
    ylabel(r'$ \operatorname{max}_{t,\phi_{\mathrm{arrival}},\psi_{\mathrm{template}}} \langle s | h \rangle$')
    title('max = %f'%max(matchlist) )
    savefig( osjoin(diagnostics_dir,'effect_of_template_orbital_phase.pdf'), pad_inches=1 )
    close(gcf())


    # Effect of signal polarization on match
    print ' * '+yellow('Effect of signal polarization on match')
    sprange = linspace(0,pi,61)
    splist = array( map( lambda sp: mo.calc_template_pol_optimized_match( signal_polarization = sp ), sprange ) )
    figure( figsize = 1.5*array([6,4]) )
    plot( sprange, splist, 'o', color=0.5*ones(3), mfc='none' )
    spo_ = linspace(0,pi,2e2)
    plot( spo_, spline(sprange,splist)(spo_), '-b' )
    axvline( pi/4, color='r', linestyle='--' )
    axvline( 3*pi/4, color='r', linestyle='--' )
    xlabel(r'$\psi_{\mathrm{signal}}$ (Signal Polarization)')
    xlim(lim(spo_))
    ylabel(r'$ \operatorname{max}_{t,\phi,\psi_{\mathrm{signal}}} \langle s | h \rangle$')
    title(r'Optimized over $\psi_{\mathrm{template}}$')
    savefig( osjoin(diagnostics_dir,'effect_of_signal_polarization_on_match.pdf'), pad_inches=1 )
    close(gcf())


    # Effect of Signal Polarization on Signal Optimal SNR
    print ' * '+yellow('Effect of signal solarization on signal optimal SNR')
    sprange = linspace(0,pi,29)
    def spf(sp):
        return mo.apply( signal_polarization = sp ).signal['optimal_snr']
    splist = array( map( spf, sprange ) )
    figure()
    plot( sprange, splist, '-b', marker='o', mfc='none' )
    spo_ = linspace(0,pi,2e2)
    plot( spo_, spline(sprange,splist)(spo_), '-k', alpha=0.4 )
    xlabel(r'$\psi_{\mathrm{signal}}$ (Signal Polarization)')
    xlim(lim(spo_))
    ylabel(r'$\rho_{\mathrm{opt}}$')
    savefig( osjoin(diagnostics_dir,'effect_of_signal_polarization_on_optSNR.pdf'), pad_inches=1 )
    close(gcf())

    # Pad teh bulleted list
    print ''

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Caclulate matches and moments over inclinations
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #

    #
    alert('Caclulating matches and moments over inclinations',heading=True)
    match_info = mo.calc_match_sky_moments( signal_wfarr_fun,
                                            template_wfarr_fun,
                                            N_theta = config.N_inclinations,
                                            N_psi_signal = config.N_PSI_signal,
                                            N_phi_signal = config.N_phi_signal,
                                            hm_vs_quad = True,
                                            verbose = True )


    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Plot results and output data
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ #
    # Setup plotting backend
    mpl.rcParams['lines.linewidth'] = 0.8
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 26
    mpl.rcParams['axes.titlesize'] = 20
    #
    alert('Ploting results and outputting data',heading=True)
    # Output all match information
    filepath = data_dir + 'match_info.pickle'
    with open(filepath, 'wb') as datafile:
        pickle.dump( match_info , datafile, pickle.HIGHEST_PROTOCOL )
    # Output match object and waveform data
    filepath = data_dir + 'misc.pickle'
    with open(filepath, 'wb') as datafile:
        pickle.dump( (signal_wfarr_fun,template_wfarr_fun,M_Sol,D_Mpc,phys_template, phys_signal) , datafile, pickle.HIGHEST_PROTOCOL )
    # Output workflow settings
    filepath = data_dir + 'workflow_settings.pickle'
    with open(filepath, 'wb') as datafile:
        pickle.dump( config, datafile, pickle.HIGHEST_PROTOCOL )
    for k in match_info:
        if isinstance(match_info[k],(ndarray,list,tuple)):
            savetxt( data_dir+k+'.asc', match_info[k] )
    figure( figsize = 6*array([1.5,1]) )
    alpha = 0.05
    sth = linspace(0,pi)
    sm = lambda x: spline( match_info['theta'], match_info[x], k=2 )(sth)
    fill_between( sth, sm('min'), sm('max'), color='k', alpha=alpha, edgecolor='none' )
    fill_between( sth, sm('quadrupole_min'), sm('quadrupole_max'), color='k', alpha=alpha, edgecolor='none' )
    plot( sth, sm('weighted_avg'),'-k',label='PhenomHM' )
    plot( sth, sm('quadrupole_weighted_avg'),'--k',label='PhenomD' )
    hline = [0.97,0.98,0.99,1.0]
    for val in hline:
        axhline(val,linestyle=':',color='k',alpha=0.5)
    xlim([0,pi])
    legend( frameon=False, loc=4 )
    xtk_labels =    [ '0',  r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$' ]
    xtk = pi*array( [  0,   1.0/4,      1.0/2,      3.0/4,       1  ] )
    xticks(xtk,xtk_labels)
    gca().tick_params(axis='y', which='major', labelsize=18)
    gca().tick_params(axis='x', which='major', labelsize=20)
    xlim( [0,pi] )
    xlabel(r'$\iota$')
    ylabel(r'$( h_\mathrm{HM} | h_\mathrm{NR} )$')
    yl = lim( hstack([sm('min'), sm('max'), sm('quadrupole_min'), sm('quadrupole_max')]) ); dy = 0.001*yl
    ylim( yl + dy*array([-1,1]) )
    savefig( basedir+'matches_'+sce.simname+'.pdf' )
    close(gcf())


    #
    import corner

    #
    X = array( [ match_info['samples']['optsnr'],
                 match_info['samples']['phi_signal'],
                 match_info['samples']['psi_signal'],
                 match_info['samples']['theta'],
                 match_info['samples']['match']] ).T

    c = corner.corner( X, range=[ lim(X[:,k],dilate=0.05) for k in range(X.shape[-1]) ],
                       bins=50,quantiles=[0.16, 0.5, 0.84],
                       labels=[ r'$\rho_{\mathrm{opt}}$',
                                r'$\phi_s$',
                                r'$\psi_s$',
                                r'$\iota_s$',
                                r'$\langle s | h_\mathrm{HM} \rangle$' ],
                       show_titles=True )
    c.set_size_inches( 13*array([1,1]) )
    savefig( basedir+'matches_'+sce.simname+'_corner.pdf',bbox_inches='tight' )

    #
    alert('Done!!!!',header=True)
