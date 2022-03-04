#!/usr/bin/bash

source /usr/workspace/cv_ddmd/.radical/auth

which python
which gcc
which radical-stack
hostname

radical-stack
 
python driver_rmsd.py --output_dir $1 --nodes $2 --walltime $3 --compute_zcentroid $4


