#
from nrutils.core import global_settings
from nrutils import scentry
from os.path import dirname, basename, isdir, realpath, abspath, join, splitext, isfile
from os import pardir,system,popen
import pickle
import glob
from nrutils.core.basics import *
from nrutils.core.nrsc import scconfig
from commands import getstatusoutput as bash

# system('clear')

# Targeted database to update and the location of the new simulation files
db_to_update = 'hogshead'
new_sim_path = '/mnt/hogshead/NR_data/ReducedData-hogshead/q4/120_point/q4a04t60dPm1_T_120_480'

# Code adopted from scbuild function in nrutils.core

# Look up metadata and config parameters
cpath_list = glob.glob( global_settings.config_path+'*.ini' )

# Filter the available .ini files for the desired database
if isinstance(db_to_update,(str,unicode)):
    msg = 'Filtering ini files for \"%s\"'%cyan(db_to_update)
    alert(msg)
    cpath_list = filter( lambda path: db_to_update in path, cpath_list )

#
if not cpath_list:
    msg = 'Cannot find configuration files (*.ini) in %s' % global_settings.config_path
    error(msg)


# Create config object from the config file (assuming only one .ini file per database, take the first!)
config = scconfig( cpath_list[0] )

# Create streaming log file
logfstr = global_settings.database_path + '/' + splitext(basename(config.config_file_location))[0] + '.log'
msg = 'Opening log file in: '+cyan(logfstr)
alert(msg)
logfid = open(logfstr, 'w')

# Grabbing current database
current_database_path = global_settings.database_path + splitext(basename(config.config_file_location))[0] + '.' + global_settings.database_ext
if isfile(current_database_path):
    msg = 'Opening database file in: '+cyan(current_database_path)
    alert(msg)
    with open( current_database_path, 'r') as data_file:
        current_database = pickle.load(data_file)
else:
    msg = "Database doesn't exist!"+cyan(current_database_path)
    error(msg)


# Generate a list of current simulation directories in the database
current_db_dirs = [e.raw_metadata.source_dir[0] for e in current_database ]
current_db_simnames = [e.simname for e in current_database ]

# Check that the desired simulation (to be added) isn't already in the database
if new_sim_path in current_db_dirs:
    msg = 'The "new" simulation directory is already in the database!'
    alert(msg)
    quit()


# Search for the simulation file with the correct metadata_id
msg = 'Searching for %s in %s.' % ( cyan(config.metadata_id), cyan(new_sim_path) ) + yellow(' This may take a long time if the folder being searched is mounted from a remote drive.')
alert(msg)
mdfile_list = rfind(new_sim_path,config.metadata_id,verbose=True)
alert('done.')

# Attempt scentry file creation from mdfiles
for mdfile in mdfile_list:
    # Create scentry object
    entry = scentry(config,mdfile,verbose=True)

    # write to log file
    logfid.write( '%5i\t%s\n'% (0,entry.log) )

    if entry.isvalid:
        # entry is valid and now we can check if it's already in the database via simname
        if entry.simname in current_db_simnames:
            msg = 'The "new" simulation name is already in the database!'
            alert(msg)
            break
        else:
            # If the obj is valid, add it to the catalog list
            alert('Simulation missing in the database and will be added\t\t\t\t %s'%yellow(entry.simname))

            # Backup old database
            db_backup_path = current_database_path + '.backup'
            msg = 'Creating backup of database file to %s'%cyan(db_backup_path)
            alert(msg,'scbuild')
            with open(db_backup_path, 'wb') as dbf:
              pickle.dump( current_database , dbf, pickle.HIGHEST_PROTOCOL )

            # Append new entries to the old database
            new_database = current_database[:]
            new_database.append(entry)

            # Write updated database to file
            msg = 'Saving updated database file to %s'%cyan(current_database_path)
            alert(msg,'scbuild')
            with open( current_database_path, 'wb') as data_file:
                pickle.dump( new_database, data_file, pickle.HIGHEST_PROTOCOL )
    else:
        # .bbh file not valid to create scentry object
        msg = 'This entry is not valid: {}'.format(mdfile)
        alert(msg)


# Close the log file
logfid.close()
