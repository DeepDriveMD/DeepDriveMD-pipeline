#!/usr/bin/bash

source /usr/workspace/cv_ddmd/.radical/auth

which python
which gcc
which radical-stack
hostname

radical-stack

python driver_embeddings.py -o $1 -n $2 -w $3 -z $4


