'''

Key reference:
https://dcc.ligo.org/DocDB/0123/T1500606/002/NRInjectionInfrastructure.pdf

-- lionel.london@ligo.org 2016 --

'''

# Import useful things
from os import system, remove, makedirs, path
from os.path import dirname, basename, isdir, realpath
from numpy import arccos as acos
from numpy import array,ones,pi,loadtxt,hstack,dot,zeros,cross
from numpy.linalg import norm
from matplotlib.pyplot import *
from os.path import expanduser
# Import needed libs
system('clear')

#
from nrutils import alert,nr2h5,scsearch,gwylm,alert
from nrutils.tools.unit.conversion import *

#
this_script = 'nr2h5_example'

#
HACK = False

# Search for simulations: Use the CFUIB high resolution base case
alert('Finding NR simulation catalog objects for realted HDF5 creation. This script is specialized to  work with BAM data.',this_script )
A = scsearch(keyword='base_96',verbose=True) # base_96 # q1.2_dc2dcp2 # q1.2_dc1dc2

# Extraction radius found using the "r" parameter in the realted config file for bam runs as well as a mapping of this to the actual extration radius as given by the bbh metadata files.
alert('Manually defining extration radius to use for cropping of NR data. This is realted to the extration parameter in the institute''s config file, and allows the calculation of the retarded time, t_retarded = t + extraction_radius',this_script )
extraction_radius = 140

#
alert('Load and crop all waveforms. Waveforms will start at the after_junkradiation_time noted in the .bbh metadata',this_script )
for a in A:

    # Convert a single simulation into a waveform object with desired multipoles
    y = gwylm( scentry_obj = a, lmax=5, verbose=True, w22 = a.raw_metadata.freq_start_22 )

    # Crop initial junk radiation from waveform without smooth windowing to be consistent with the infrastructure's conventions
    y.clean( method='crop', crop_time=float(a.raw_metadata.after_junkradiation_time)+extraction_radius  )

    # # plot the waveform(s)
    # y.plot(show=True,kind='strain')

    # ------------------------------------------------------- #
    # Parameters needed to make hdf5 file.
    # ------------------------------------------------------- #
    alert('Defining parameters needed to make hdf5 files from runs: metadata, file io strings',this_script )

    # Name of hdf5 file to create, includes file extension
    if a.simdir()[-1] == '/':
        run_label = a.simdir().split('/')[-2]
    else:
        run_label = a.simdir().split('/')[-1]

    # If using the hacked version, then add a tag to the output file name
    if HACK:
        run_label = 'hacked_' + run_label

    # Where to save hdf5 file
    output_path = '/Users/book/JOKI/Libs/KOALA/nrutils_dev/review/data/%s.h5' % run_label

    # Define universal time data to be the max of the sum of the l=2 multipoles^2
    alert('Creating universal time series using sum of l=2 multipoles.',this_script )
    universal_amp =   [ k for k in y.hlm if (k.m==2 and k.l==2) ][0].amp**2 \
                    + [ k for k in y.hlm if (k.m==1 and k.l==2) ][0].amp**2 \
                    + [ k for k in y.hlm if (k.m==0 and k.l==2) ][0].amp**2 \
                    + [ k for k in y.hlm if (k.m==-1 and k.l==2) ][0].amp**2 \
                    + [ k for k in y.hlm if (k.m==-2 and k.l==2) ][0].amp**2
    # Seed universal time array with the l=m=2 time
    universal_t = [ k for k in y.hlm if (k.m==2 and k.l==2) ][0].t
    # Center universal time about peak
    universal_t -= universal_t[ list(universal_amp).index( max(universal_amp) ) ]
    # Name the l=m=2 strain for use later
    h22 = [ k for k in y.hlm if (k.m==2 and k.l==2) ][0]

    # Create dictionary of mode coordinates and waveform data
    alert('Creating dictionary of strain multipoles',this_script )
    nr_strain_data = {}
    for hlm in y.hlm:
        # NOTE that there's a -1 factored into the phase as the NR infrastructure uses the opposite sign convention compared to nrutils
        nr_strain_data[  ( hlm.l, hlm.m )  ] = { 'amp':hlm.amp, 'phase': -1.0*hlm.phi, 't':universal_t }

    #
    alert('Creating metadata input for nr2h5',this_script )
    Lhat = ( y.L1 + y.L2 )   / norm( y.L1 + y.L2 )
    # NOTE: See above equation 19 in the reference pdf for the convention used here for nhat
    nhat = ( y.R1 - y.R2 ) / norm( y.R1 - y.R2 )
    # Define attributes
    nr_meta_data = {}
    nr_meta_data['NR-group'] = y.config.institute
    nr_meta_data['type'] = y.label
    nr_meta_data['name'] = y.setname
    nr_meta_data['object1'] = 'BH'
    nr_meta_data['object2'] = 'BH'
    nr_meta_data['mass1'] = y.m1
    nr_meta_data['mass2'] = y.m2
    nr_meta_data['eta'] = y.m1*y.m2 / (y.m1+y.m2)**2

    # Check required mass convention
    if nr_meta_data['mass1']<nr_meta_data['mass2']:
        raise ValueError('Mass1 must be GREATER THAN mass2.')

    # Writing spins
    alert('Writing Dimensionless spins in the correct frame.',this_script )
    X1= y.S1 / (y.m1**2)
    X2 = y.S2 / (y.m2**2)
    # Define function to rotate spin ACCORDING TO EQS 9-11
    def RotateSpin(X):
        X_LAL = zeros( X.shape )
        X_LAL[0] = dot( X, nhat )
        X_LAL[1] = dot( X, cross(Lhat,nhat) )
        X_LAL[2] = dot( X, Lhat )
        return X_LAL
    # Rotate spins
    X1_LAL = RotateSpin( X1 )
    X2_LAL = RotateSpin( X2 )
    # Store spin values
    nr_meta_data['spin1x'] = X1[0]
    nr_meta_data['spin1y'] = X1[1]
    nr_meta_data['spin1z'] = X1[2]
    #
    nr_meta_data['spin2x'] = X2[0]
    nr_meta_data['spin2y'] = X2[1]
    nr_meta_data['spin2z'] = X2[2]
    #
    nr_meta_data['LNhatx'] = Lhat[0]
    nr_meta_data['LNhaty'] = Lhat[1]
    nr_meta_data['LNhatz'] = Lhat[2]
    nr_meta_data['nhatx'] = nhat[0]
    nr_meta_data['nhaty'] = nhat[1]
    nr_meta_data['nhatz'] = nhat[2]

    #
    if True is HACK:
        msg = red('Warning:')+yellow(' Forcing the appearance of LAL convention for the separation vector and angular momentum unit vector.')
        alert(msg,'nr2h5_example')
        nr_meta_data['LNhatx'] = 0.0
        nr_meta_data['LNhaty'] = 0.0
        nr_meta_data['LNhatz'] = 1.0
        nr_meta_data['nhatx'] = 1.0
        nr_meta_data['nhaty'] = 0.0
        nr_meta_data['nhatz'] = 0.0

    #
    print green('## ') + 'Lhat = (%f.%f,%f)' % tuple(Lhat)
    print green('## ') + 'nhat = (%f.%f,%f)' % tuple(nhat)
    print green('## ') + 'm1   = %f' % y.m1
    print green('## ') + 'm2   = %f' % y.m2
    print green('## ') + green( 'X1_LAL = (%f.%f,%f)' % tuple(X1_LAL ) )
    print green('## ') + green( 'X2_LAL = (%f.%f,%f)' % tuple(X2_LAL ) )
    print green('## ') + 'X1   = (%f.%f,%f)' % tuple(X1)
    print green('## ') + 'X2   = (%f.%f,%f)' % tuple(X2)

    # Write store the projected spins to the meta-data dictionary for ease of reference
    nr_meta_data['spin1x_lal'] = X1_LAL[0]
    nr_meta_data['spin1y_lal'] = X1_LAL[1]
    nr_meta_data['spin1z_lal'] = X1_LAL[2]
    #
    nr_meta_data['spin2x_lal'] = X2_LAL[0]
    nr_meta_data['spin2y_lal'] = X2_LAL[1]
    nr_meta_data['spin2z_lal'] = X2_LAL[2]

    # nr_meta_data['f_lower_at_1MSUN'] = physf( y.raw_metadata.freq_start_22/(2.0*pi) , 1.0 ) # here the "1" is for 1 solar mass
    nr_meta_data['f_lower_at_1MSUN'] = physf( h22.dphi[0]/(2.0*pi) , 1.0 ) # here the "1" is for 1 solar mass
    print '>> Old f_lower_at_1MSUN = %f' % physf( y.raw_metadata.freq_start_22/(2.0*pi) , 1.0 )
    print '>> New f_lower_at_1MSUN = %f' % nr_meta_data['f_lower_at_1MSUN']
    nr_meta_data['eccentricity'] =  y.raw_metadata.eccentricity
    nr_meta_data['PN_approximant'] = 'None'
    nr_meta_data['coa_phase'] = acos( dot( nhat,[0,1,0] ) )


    # ------------------------------------------------------- #
    # Call nr2h5 to make hdf5 file
    # ------------------------------------------------------- #
    alert('Creating HDF5 files:',this_script )
    #nr2h5( nr_strain_data, nr_meta_data, output_path=output_path, verbose=True )
