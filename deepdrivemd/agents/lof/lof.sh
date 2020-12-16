#!/bin/bash

cmd_params=$@

export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

echo ${cmd_params}
python3 /path/to/optics.py $cmd_params --distributed
