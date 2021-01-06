#!/bin/bash

cmd_params=$@

# important variables
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_PORT=29500
export MASTER_ADDR=$(cat ${LSB_DJOB_HOSTFILE} | uniq | sort | grep -v batch | head -n1)
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
export WANDB_MODE=dryrun

# determine gpu
enc_gpu=$(( ${LOCAL_RANK} ))

# launch code
cmd="$cmd_params -E ${enc_gpu} --distributed"
echo ${cmd}
($cmd)