#!/bin/bash
#
# Clean up all the junk from previous runs to set the stage for a new run
#
# - Remove the pilot sandbox
rm -rf /lustre/orion/scratch/hjjvd/chm136/radical.pilot.sandbox/re.session.login?.hjjvd.*
# - Remove the client sandbox
rm -rf re.session.login?.hjjvd.*
# - Remove the scientific data directory
rm -rf /lustre/orion/scratch/hjjvd/chm136/test_nwchem
# - Remove the RADICAL analytics cache
rm -rf /ccs/home/hjjvd/.radical/analytics/cache/re.session.login?.hjjvd.*.pickle
# - Remove the data analysis artifacts
rm -f  ../postproduction/*.csv ../postproduction/*.png
