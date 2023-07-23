source /usr/workspace/cv_ddmd/.radical/auth

which python
which gcc
which radical-stack
hostname

radical-stack

# HDF5 often fails to lock files on compute nodes
# so best turn locking off to prevent failures
# in the reporter.
#
# The error message in question is:
# "Unable to create file (unable to lock file, errno = 524, error message = 'Unknown error 524')"
#
# More information on this error:
# https://github.com/nanoporetech/medaka/issues/240
#
# See for the reporter h5py code: 
# - MD-tools/mdtools/openmm/reporter.py
# - MD-tools/mdtools/nwchem/reporter.py
export HDF5_USE_FILE_LOCKING="FALSE"
 
#python  -m deepdrivemd.deepdrivemd_stream -c bba/$1/config.yaml
python  -m deepdrivemd.deepdrivemd -c bba/$1/config.yaml

