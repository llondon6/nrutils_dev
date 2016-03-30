#!/usr/bin/python
'''
 sxs_download.py

 This script is to download waveforms from the open SXS collaboration website.

 ~ llondon2'14

'''


# Imported goods
from nrutils import *

# Load the settings file
settings = smart_object()

# NOTE that the line below is NUIX specific
settings_file_location = os.path.dirname(os.path.realpath(__file__))+'/'+os.path.basename(__file__).split('.')[0]+'.ini'

settings.learn_file(settings_file_location)


# Clear the prompt
os.system('clear')

#--------------------------------------------------------------------------------- #
# Alert the user about this program
# --------------------------------------------------------------------------------- #
wave_train = '~~~~<vvvvvvvvvvvvvWw>~~~~~~~'
print_hl(wave_train)
center_print('')
program_title = 'SXSDLTOOL (BETA)'
print( center_space(program_title) + bold(program_title) )
center_print('A Python tool for the automated downloading of publicly available graviational wave data from the SXS Collaboration.')
author_info = 'Author: Koala Bear, Email: koalascript@gmail.com'
print( center_space(author_info)+bold(author_info) )
center_print('About SXS: black-holes.org')
center_print('')
print_hl(wave_train)
print('')
time.sleep(1)

# Print settings to screen
print_hl()
print( '>> '+bold('Settings file loaded:') )
print_hl()
settings.show()
time.sleep(4)

# --------------------------------------------------------------------------------- #
# For all SXS simulations available, download the metadatafiles into a special folder
# --------------------------------------------------------------------------------- #

# Get user's home directory
from os.path import expanduser
home = expanduser("~")

# # Name location where metadafiles will be stored
# work_dir = home+'/WORK/'
# mkdir(work_dir) # Note, this is a custom mkdir function
# # High level data storage directory
# data_dir = work_dir+'Data/'
# mkdir(data_dir)
# # Parent directory for waveform sets
# wave_dir = data_dir+'Waveforms/'
# mkdir(wave_dir)

# Get base directory from settings file
wave_dir = settings.BASE_DIRECTORY
if wave_dir[-1] is not '/':
    wave_dir = wave_dir + '/'

# Location of Catalog file
catalog_dir = wave_dir+'Catalog/'
mkdir(catalog_dir)
# Location where waveforms and metadata will be stores
sxs_dir = wave_dir+'SXS/'
mkdir(catalog_dir)

# --------------------------------------------------------------------------------- #
# For all SXS simulations available, download the metadatafiles into a special folder
# --------------------------------------------------------------------------------- #

# Alert the user that matadata dl is starting
print '>> '+bold('Starting the download of Meta Data.')

catalog_path = get_metadata(sxs_dir,catalog_dir,verbose=True,replace_old=settings.REPLACE_OLD)

# Read the initial parameters from the meta data and test against desired inputs
search_results = sc_search( verbose=True, nonspinning=settings.NON_SPINNING,\
                            SET_DEFAULT_CATALOG_LOCATION=catalog_path )

# search_results = sc_search( verbose=True, nonspinning=settings.NON_SPINNING )

# --------------------------------------------------------------------------------- #
# For passing meta data, create the relevant simulation folder(s) and download the simlation h5 file structure
# --------------------------------------------------------------------------------- #

# Ensure that what to downlod is an iterable list
if type(settings.WHAT_TO_DOWNLOAD) is str:
    settings.WHAT_TO_DOWNLOAD = [settings.WHAT_TO_DOWNLOAD]

# Define the string format used for each file
for server_file in settings.WHAT_TO_DOWNLOAD:

    # Define the url location format that will be used for each download
    url_format = 'http://www.black-holes.org/waveforms/data/Download.php/?id=SXS:BBH:%04i&file=Lev%i/'+server_file+'.h5'

    # Define a prefix for data files
    data_prefix = server_file.split('_')[0] + '_'

    # Loop through resutls of sc_search
    for Y in search_results:

        # Get this run's ID
        run_id = Y.id

        # Get the simulation's resolution level -- this is the highest resolution by default
        Level = int( Y.simulation_name.split('Lev')[-1] )

        # Use the counter and Level varibale to create a url
        url = url_format % (run_id,Level)

        # Use the inctrement to uniquely name each file
        file_name = 'sxs%04i.h5.tar' % run_id

        # Only download if the decompressed file does not already exist
        run_dir = Y.source_dir[0]

        # Define smallest allowed file size
        min_bytes = 200

        # This is the expected content of the tar file. We only know this becuase of prior experience.
        data_name = url_format.split('/')[-1]

        #
        status_file = run_dir + '.status'
        download_state = smart_object()
        if not os.path.isfile(status_file):
            f = open( status_file, 'w' )
            f.write('\n')
            f.write('########################################################################\n')
            f.write('# 1=True if sxsdltool has finished with this folder, 0=False otherwise #\n')
            f.write('########################################################################\n')
            f.write('STATUS = 0\n')
            f.close()

        #
        download_state.learn_file(status_file)

        # Only proceed with download if this directory has not been passed before
        if not download_state.STATUS:

            # Determine whether the download is desired
            file_location = run_dir+data_name
            print('>> Querying "%s"' % file_location)
            DOWNLOAD = os.path.isfile(file_location) and settings.REPLACE_OLD
            DOWNLOAD = DOWNLOAD or not os.path.isfile(file_location)

            if DOWNLOAD:
                # Download the tar file
                print('>> Downloading TAR-FILE for:\t"%s"' % url )
                empty,tar_location = download(url,run_dir,save_name=file_name,\
                                            verbose=True, size_floor=min_bytes)
                # Decompress
                print('>> Decrompressing "%s"' % run_dir+file_name )
                untar(tar_location,cleanup=True)
            else:
                # Let the user know that this download has been skipped
                print('>> '+bold('Skipping download. ')+'Data file already exists.')

            # Extract desired information form h5 file, then remove everything else to preserve disk space
            h5_file_string = run_dir + data_name
            h5tofiles( h5_file_string, run_dir, file_filter = lambda s: l_test(s,settings.L_MAX), \
                        cleanup = settings.DELETE_H5_FILES, prefix = data_prefix )

            # Mark thisdirectory as complete
            pattern = "STATUS"
            substitute = "STATUS = 1"
            replace_line(status_file, pattern, substitute)

        else:

            #
            print(bold('>> Status found to be complete. Moving on.'))
