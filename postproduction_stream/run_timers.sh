#!/usr/bin/bash

source /usr/workspace/cv_ddmd/.radical/auth

which python
which gcc
which radical-stack
hostname

radical-stack
 
python driver_timers.py --output_dir $1 --nodes $2 --walltime $3 --session $4 --pilot $5 --exclude $6


