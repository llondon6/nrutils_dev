
#
from nrutils.core.basics import *
from glob import glob as ls
from os.path import getctime
from numpy import array,cross,zeros,dot,abs,sqrt,inf,nan,sign
from numpy.linalg import inv, norm
from numpy import sum as asum

# Determine whether the folder containing a metadta file is valid: can it be used to reference waveform data?
def validate( metadata_file_location, config = None, verbose = True ):

    #
    from os.path import isfile as exist
    from os.path import abspath,join,basename
    from os import pardir

    #
    run_dir = abspath( join( metadata_file_location, pardir ) )+'/'

    # The folder is valid if there is l=m=2 mode data in the following dirs
    status = len( ls( run_dir + '/Ylm_WEYLSCAL4::Psi4r_l2_m1_r*' ) ) > 0
    status = status or len( ls( run_dir + '/mp_WeylScal4::Psi4i_l2_m2_r*' ) ) > 0

    # Let the people know
    if not status:
        msg = 'waveform data could not be found.'
        if verbose: warning(msg,'maya.validate')

    # ignore directories with certain tags in filename
    ignore_tags = ['backup','old','archive']
    for tag in ignore_tags:
        status = status and not ( tag in run_dir )

    # ensure that file name is the same as the folder name
    a = basename(metadata_file_location).split(config.metadata_id)[0]
    b = parent(metadata_file_location)
    status = status and (  a in b  )

    #
    return status


#
def learn_metadata( metadata_file_location ):

    #
    thisfun = 'maya.learn_metadata'

    # Look for stdout files
    stdout_file_list = sorted( ls( parent(metadata_file_location)+'/stdout*' ) )
    if not stdout_file_list:
        msg = 'cannot find stdout files which contain important metadata'
        error(msg,'maya.learn_metadata')
    # Look for ShiftTracker files
    shift_tracker_file_list = ls( parent(metadata_file_location)+'/Shift*' )
    if not shift_tracker_file_list:
        msg = 'cannot find ShiftTracker* files which contain information about binary dynamics'
        error(msg,'maya.learn_metadata')
    # Look for Horizon mass and spin files
    hn_file_list = ls( parent(metadata_file_location)+'/hn*' )
    if not hn_file_list:
        msg = 'cannot find hn_masspin files which contain information about remnant BH final state'
        error(msg,'maya.learn_metadata')

    # Use the first file returned by the OS
    # NOTE that this file is neeeded to get the component and net ADM masses
    stdout_file_location = stdout_file_list[0]

    # Learn the par file
    raw_metadata = smart_object( metadata_file_location )

    # Shortand
    y = raw_metadata

    # Retrieve ADM masses form the stdout file
    y.learn_string( grep( 'm+', stdout_file_location )[0] )
    y.learn_string( grep( 'm-', stdout_file_location )[0] )
    y.learn_string( grep( 'ADM mass from r', stdout_file_location )[0] )

    # # Useful for debugging -- show what's in y
    # y.show()

    # Create smart_object for the standard metadata
    standard_metadata = smart_object()
    # shorthand
    x = standard_metadata

    # Creation date of metadata file
    x.date_number = getctime(  metadata_file_location  )

    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Calculate derivative quantities  %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    # Masses
    x.m1 = getattr(y,'INFO(TwoPunctures):ADMmassforpuncture:m+')
    x.m2 = getattr(y,'INFO(TwoPunctures):ADMmassforpuncture:m_')

    #
    P1 = array( [ getattr(y,'twopunctures::par_P_plus[0]'),
                  getattr(y,'twopunctures::par_P_plus[1]'),
                  getattr(y,'twopunctures::par_P_plus[2]') ] )
    P2 = array( [ getattr(y,'twopunctures::par_P_minus[0]'),
                  getattr(y,'twopunctures::par_P_minus[1]'),
                  getattr(y,'twopunctures::par_P_minus[2]') ] )

    #
    S1 = array( [ getattr(y,'twopunctures::par_s_plus[0]'),
                  getattr(y,'twopunctures::par_s_plus[1]'),
                  getattr(y,'twopunctures::par_s_plus[2]') ] )
    S2 = array( [ getattr(y,'twopunctures::par_s_minus[0]'),
                  getattr(y,'twopunctures::par_s_minus[1]'),
                  getattr(y,'twopunctures::par_s_minus[2]') ] )

    # Read initial locations from the ShiftTracker files
    def shiftt_to_initial_bh_location(key):
        shiftt0_file_location = [ f for f in shift_tracker_file_list if key in f ][0]
        fid = open( shiftt0_file_location )
        #return array( [ float(a) for a in fid.readline().replace('\n','').split('\t')][2:5] )
        done = False
        while not done:
            raw_line = fid.readline()
            done = raw_line[0] != '#'
        #
        line = raw_line.replace('\n','').split('\t')
        # get x y z from columns of first non-commented line
        ans = array( [ float(a) for a in line][2:5] )
        return ans
    R1 = shiftt_to_initial_bh_location("ShiftTracker0")
    R2 = shiftt_to_initial_bh_location("ShiftTracker1")

    # Find initial binary separation for convenience
    x.b = norm(R1-R2)

    #
    x.note = 'Binary parameters correspond to initial data, not an after-junk point.'

    # Estimate the component angular momenta
    L1 = cross(R1,P1)
    L2 = cross(R2,P2)

    # Extract and store the initial adm energy
    x.madm = getattr(y,'INFO(TwoPunctures):ADMmassfromr')[-1]

    # Store the initial linear momenta
    x.P1 = P1; x.P2 = P2
    x.S1 = S1; x.S2 = S2

    # Estimate the initial biary separation (afterjunk), and warn the user if this value is significantly different than the bbh file
    x.b = norm(R1-R2)

    #
    x.R1 = R1; x.R2 = R2

    #
    x.L1 = L1; x.L2 = L2

    #
    L = L1+L2
    S = S1+S2
    x.L = L
    x.J = L+S

    # Load Final Mass and Spin Data  hn_mass_spin_2
    hn_file_bin = [ f for f in hn_file_list if 'hn_mass_spin_2' in f ]
    proceed = len(hn_file_bin)==1
    hn_file = hn_file_bin[0] if proceed else None
    nan_remnant_data = array( [ nan,nan,nan,nan,nan ] )

    #
    if not proceed:
        #
        msg = 'The default hn_mass_spin_2 file could not be found. Place-holders (i.e. nans) for the remnant information will be passed.'
        warning(msg,thisfun)
        x.note += ' ' + msg
        remnant_data = nan_remnant_data
    else:
        # Use bash's tail to get the last row in the file
        cmd = 'tail -n 1 %s' % hn_file
        data_string = bash(cmd).decode('utf-8')
        # If the string is empty, then there was an error
        if not data_string:
            msg = 'The system failed using tail to get the remnant state from \"%s\"'%cyan(cmd)
            error(msg,thisfun)
        # Else, parse the data string into floats
        remnant_data = [ float(v) for v in data_string.replace('\n','').split('\t') ]
        # Handle formatting cases
        if len(remnant_data) != 5:
            msg = 'Remanant data was loaded, but its format is unexpected (last row length is %s). Placeholders for the remnant information will be passed.' % yellow( str(len(remnant_data)) )
            warning(msg,thisfun)
            x.note += ' ' + msg
            remnant_data = nan_remnant_data

    # Unpack the remnant data
    [tf,Mf,xfx,xfy,xfz] = remnant_data
    # Store related final mass and spin data
    x.mf = Mf
    x.Sf = Mf*Mf*array([xfx,xfy,xfz])
    x.Xf = array([xfx,xfy,xfz])
    x.xf = sign(x.Sf[-1])*norm(x.Sf)/(x.mf*x.mf)

    # Store relaxed (after-junk) fields
    x.S1_afterjunk,x.S_afterjunk2,x.S_afterjunk = None,None,None
    x.L1_afterjunk,x.L2_afterjunk,x.L_afterjunk = None,None,None
    x.R1_afterjunk,x.R2_afterjunk = None,None
    x.P1_afterjunk,x.P2_afterjunk = None,None
    x.J_afterjunk = None
    
    #
    x.S = x.S1 + x.S2
    x.L = x.L1 + x.L2
    x.J = x.L + x.S

    #
    x.valid = True

    #
    return standard_metadata, raw_metadata


# Given an extraction parameter, return an extraction radius
def extraction_map( this, extraction_parameter, verbose=False ):

    # Given an extraction parameter, return an extraction radius
    extraction_radius = extraction_parameter

    #
    return extraction_radius


# Estimate a good extraction radius and level for an input scentry object from the BAM catalog
def infer_default_level_and_extraction_parameter( this,     # An scentry object
                                                  desired_exraction_radius=None,    # (Optional) The desired extraction radius in M, where M is the initial ADM mass
                                                  verbose=None ):   # Toggel to let the people know
    '''Estimate a good extraction radius and level for an input scentry object from the BAM catalog'''

    # NOTE that input must be scentry object
    # Import useful things
    from glob import glob
    from numpy import array,argmin

    # Handle the extraction radius input
    # NOTE that the default value of X is chosen to ensure that there is always a ringdown
    desired_exraction_radius = this.config.default_par_list[0] if desired_exraction_radius is None else desired_exraction_radius

    # Find all l=m=2 waveforms
    search_string = this.simdir() + '*Psi4*l2_m2*0.asc'
    file_list = glob( search_string )

    # For all results
    exr,rad = [],[]
    for f in file_list:

        # Split filename string to find level and extraction parameter
        f.replace('//','/')
        f = f.split('/')[-1]
        parts = f.split('l2_m2_r') # e.g. "mp_WeylScal4::Psi4i_l2_m2_r75.00.asc".split('_')
        exr_ = float( parts[-1].split('.asc')[0] )
        # Also get related extraction radius (M)
        rad_ = extraction_map( this, exr_ )
        # Append lists
        exr.append(exr_);rad.append(rad_)

    # NOTE that we will use the extraction radius that is closest to desired_exraction_radius (in units of M)
    k = argmin( abs( desired_exraction_radius - array(rad) )  )
    extraction_parameter = exr[k]

    # Also store a dictionary between extraction parameter and extraction radius
    # And a dictionary between the level parameter and extraction radius.
    # For Maya this is not used and so it's set to None. BAM uses it though.
    extraction_map_dict = {}
    extraction_map_dict['radius_map'] = { exr[n]:r for n,r in enumerate(rad) }
    extraction_map_dict['level_map'] = None

    # Note that maya sims have no level specification
    level = None

    # Return answers
    return extraction_parameter,level,extraction_map_dict



#
def learn_source_dynamics(scentry_object,dynamics_times,verbose=False):

    '''
    Based on notebook by Jonathan Thompson, 2019self.

    NOTES
    ---
    * For now, only S, L and J will be calculated and stored. This is to be useful while avoiding faff due to mass ratio convetions.

    USAGE
    ---
    dict_with_source_dynamics = learn_source_dynamics(dynamics_times,verbose=True)

    '''

    # Import usefuls
    from positive import alert,smart_load,lim,spline,find
    from glob import glob as ls
    from nrutils.core.basics import straighten_wfarr
    from numpy import array, cross, linalg
    from numpy import max as npmax
    from numpy.linalg import norm
    from os.path import join
    
    # Look for ShiftTracker files
    simdir = scentry_object.simdir()
    from glob import glob as ls
    shift_tracker_file_list = ls( simdir+'Shift*' )
    
    # Read initial locations from the ShiftTracker files
    def sparse_read_shifttracker(key):
        
        #
        from numpy import array
        
        shiftt0_file_location = [ f for f in shift_tracker_file_list if key in f ][0]
        fid = open( shiftt0_file_location )
        #return array( [ float(a) for a in fid.readline().replace('\n','').split('\t')][2:5] )
        done = False
        # print '>>',shiftt0_file_location
        dt = 1
        warning('Due to large size of GT dynamics files, data are loaded with 1M resolution. Finer time spacing may be wanted and can be acheved using the dt keyword input.')
        lines = []
        while not done:
            raw_line = fid.readline()
            # print raw_line
            done = raw_line[0] != '#'
            if done:
                line = [ float(k) for k in raw_line.replace('\n','').split('\t') ]
                itt,time,x,y,z,vx,vy,vz,ax,ay,az = line
                lines.append( line )
        #
        done = False
        while not done:
            raw_line = fid.readline()
            try:
                line = [ float(k) for k in raw_line.replace('\n','').split('\t') ]
            except:
                # print '>>',raw_line
                if raw_line=='':
                    break
            trial_itt,trial_time,trial_x,trial_y,trial_z,trial_vx,trial_vy,trial_vz,trial_ax,trial_ay,trial_az = line
            #print (trial_time-time)
            if (trial_time-time) >= dt:
                itt,time,x,y,z,vx,vy,vz,ax,ay,az = line
                lines.append( line )
                #print line
                
        #
        dynamics = array(lines)
        
        #
        return dynamics

    #
    dynamics1 = sparse_read_shifttracker("ShiftTracker0")
    dynamics2 = sparse_read_shifttracker("ShiftTracker1")

    # Note that columns are sometimes defined in the ShiftTracker header files 
    # itt	time	x	y	z	vx	vy	vz	ax	ay	az
    itt,time1,x1,y1,z1,vx1,vy1,vz1,ax1,ay1,az1 = dynamics1.T
    itt,time2,x2,y2,z2,vx2,vy2,vz2,ax2,ay2,az2 = dynamics2.T
    
    # # Most likely, velocities are actually shifts and should be negated
    # vy1,vz1,ax1,ay1,az1 = [ -k for k in (vy1,vz1,ax1,ay1,az1) ]
    # vy2,vz2,ax2,ay2,az2 = [ -k for k in (vy2,vz2,ax2,ay2,az2) ]
    
    #
    mass1 = scentry_object.m1
    mass2 = scentry_object.m2
    
    # Positions
    R1_ = array( [ x1,y1,z1 ] ).T
    R2_ = array( [ x2,y2,z2 ] ).T
    
    # Momenta
    P1 = mass1 * array( [ vx1,vy1,vz1 ] ).T
    P2 = mass2 * array( [ vx2,vy2,vz2 ] ).T
    
    # Estimate the component angular momenta
    L1_ = cross(R1_,P1)
    L2_ = cross(R2_,P2)
    L_ = L1_+L2_
    
    # Time values
    L_times  = time1
    
    #
    def straighten(times,vec):
        vx,vy,vz = vec if vec.shape[0]==3 else vec.T
        varr = array( [ times,vx,vy,vz ] ).T
        varr = straighten_wfarr( varr, verbose=True )
        new_times,vx,vy,vz = varr.T
        new_vec = array( [vx,vy,vz] ).T
        return new_times, new_vec.T
        
    #
    L1_times,L1_ = straighten(L_times,L1_)
    L2_times,L2_ = straighten(L_times,L2_)
    #
    R1_times,R1_ = straighten(L_times,R1_)
    R2_times,R2_ = straighten(L_times,R2_)
    
    #
    S1_times = L1_times

    #
    abs_dr = linalg.norm( (R2_-R1_).T, axis=1 )
    r0 = 3.1
    premerger_t_max = R1_times[ find(abs_dr < r0)[0] ]
    internal_t_max = max(dynamics_times)
    
    #
    if internal_t_max<max(dynamics_times):
        dynamics_times = dynamics_times[ dynamics_times<internal_t_max ]
        
    #
    def mask( t ):
        msk = (t>=min(dynamics_times)) & (t<=max(dynamics_times) )
        return msk

    # Interpolate everything of use. Some care is taken with the spins as the data files may have sligtly different time series.
    dynamics_times = dynamics_times[ dynamics_times < max(lim(L_times)[-1],lim(S1_times)[-1]) ]
    
    #
    msk = mask( L1_times )
    L1 = array(  [ spline(L1_times[msk],l[msk])(dynamics_times) for l in L1_ ]  ).T
    msk = mask( L2_times )
    L2 = array(  [ spline(L2_times[msk],l[msk])(dynamics_times) for l in L2_ ]  ).T
    
    #
    msk = mask( R1_times )
    R1 = array(  [ spline(R1_times[msk],l[msk])(dynamics_times) for l in R1_ ]  ).T
    msk = mask( R2_times )
    R2 = array(  [ spline(R2_times[msk],l[msk])(dynamics_times) for l in R2_ ]  ).T

    # Total angular momenta
    L = L1+L2   # Spin

    #
    foo = {}
    
    # ORBITAL MOMENTA
    foo['L1'] = L1
    foo['L2'] = L2
    foo['L'] = L
    # SPIN MOMENTA
    foo['S1'] = L1*0
    foo['S2'] = L1*0
    foo['S'] = L1*0
    # TOTAL MOMENTA
    foo['J'] = L1*0
    # TRAJECTORIES
    foo['R1'] = R1
    foo['R2'] = R2

    # DYNAMICS TIMES USED
    foo['dynamics_times'] = dynamics_times
    
    #
    foo['premerger_t_max'] = premerger_t_max

    # Let's go! :D
    ans = foo
    return ans