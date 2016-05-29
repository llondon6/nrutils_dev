'''

Key reference:
https://dcc.ligo.org/DocDB/0123/T1500606/002/NRInjectionInfrastructure.pdf

-- lionel.london@ligo.org 2016 --

'''

# Import useful things
from os import system, remove, makedirs, path
from os.path import dirname, basename, isdir, realpath
from numpy import array,ones,pi,loadtxt,hstack
from numpy.linalg import norm
from matplotlib.pyplot import *
from os.path import expanduser
# Import needed libs
system('clear')

#
from nrutils import alert,nr2h5,scsearch,gwylm,alert
from nrutils.tools.unit.conversion import *

#
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#
this_script = 'nr2h5_example'

# Search for simulations: Use the CFUIB high resolution base case
alert('Finding NR simulation catalog objects for realted HDF5 creation. This script is specialized to  work with BAM data.',this_script )
A = scsearch(keyword="base_96",unique=True,verbose=True)

# Extraction radius found using the "r" parameter in the realted config file for bam runs as well as a mapping of this to the actual extration radius as given by the bbh metadata files.
alert('Manually defining extration radius to use for cropping of NR data. This is realted to the extration parameter in the institute''s config file, and allows the calculation of the retarded time, t_retarded = t + extraction_radius',this_script )
extraction_radius = 140

#
alert('Load and crop all waveforms. Waveforms will start at the after_junkradiation_time noted in the .bbh metadata',this_script )
for a in A:

    # Convert a single simulation into a waveform object with desired multipoles
    y = gwylm( scentry_obj = a, lmax=2, dt=0.4, verbose=True )

    # Crop initial junk radiation from waveform without smooth windowing to be consistent with the infrastructure's conventions
    y.clean( method='crop', crop_time=float(a.raw_metadata.after_junkradiation_time)+extraction_radius  )

# plot the waveform(s)
# y.plot(show=True,kind='strain')

# ------------------------------------------------------- #
# Parameters needed to make hdf5 file.
# ------------------------------------------------------- #
alert('Defining parameters needed to make hdf5 files from runs: metadata, file io strings',this_script )

# Name of hdf5 file to create, includes file extension
if A[0].simdir()[-1] == '/':
    run_label = A[0].simdir().split('/')[-2]
else:
    run_label = A[0].simdir().split('/')[-1]

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

# Create dictionary of mode coordinates and waveform data
alert('Creating dictionary of strain multipoles',this_script )
nr_strain_data = {}
for hlm in y.hlm:
    nr_strain_data[  ( hlm.l, hlm.m )  ] = { 'amp':hlm.amp, 'phase':-hlm.phi, 't':universal_t }

#
alert('Creating metadata input for nr2h5',this_script )
Lhat = (y.L1 + y.L2)   / norm( y.L1 + y.L2 )
nhat = ( y.R2 - y.R1 ) / norm( y.R2 - y.R1 )
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
nr_meta_data['spin1x'] = y.S1[0] / (y.m1**2)
nr_meta_data['spin1y'] = y.S1[1] / (y.m1**2)
nr_meta_data['spin1z'] = y.S1[2] / (y.m1**2)
nr_meta_data['spin2x'] = y.S2[0] / (y.m2**2)
nr_meta_data['spin2y'] = y.S2[1] / (y.m2**2)
nr_meta_data['spin2z'] = y.S2[2] / (y.m2**2)
nr_meta_data['LNhatx'] = Lhat[0]
nr_meta_data['LNhaty'] = Lhat[1]
nr_meta_data['LNhatz'] = Lhat[2]
nr_meta_data['nhatx'] = nhat[0]
nr_meta_data['nhaty'] = nhat[1]
nr_meta_data['nhatz'] = nhat[2]
nr_meta_data['f_lower_at_1MSUN'] = physf( y.wstart, 1 ) # here the "1" is for 1 solar mass
nr_meta_data['eccentricity'] =  y.raw_metadata.eccentricity
nr_meta_data['PN_approximant'] = 'None'


# ------------------------------------------------------- #
# Call nr2h5 to make hdf5 file
# ------------------------------------------------------- #
alert('Creating HDF5 files:',this_script )
nr2h5( nr_strain_data, nr_meta_data, output_path=output_path, verbose=True )
