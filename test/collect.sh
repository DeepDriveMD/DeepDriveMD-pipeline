#!/bin/bash
#
# Collect all the important artifacts from a previous run and wrap them up in a tar-ball
#
use_case=$1
if [[ -d /lustre/orion/scratch/hjjvd/chm136/${use_case} ]];
then
    echo "Directory /lustre/orion/scratch/hjjvd/chm136/${use_case} already exists!!!"
    echo "Cowardly giving up."
    exit 10
fi
mkdir /lustre/orion/scratch/hjjvd/chm136/${use_case}
mkdir /lustre/orion/scratch/hjjvd/chm136/${use_case}/radical.pilot.sandbox
mkdir /lustre/orion/scratch/hjjvd/chm136/${use_case}/radical.client.sandbox
cp -a /lustre/orion/scratch/hjjvd/chm136/radical.pilot.sandbox/re.session.login?.hjjvd.* /lustre/orion/scratch/hjjvd/chm136/${use_case}/radical.pilot.sandbox/.
cp -a re.session.login?.hjjvd.* /lustre/orion/scratch/hjjvd/chm136/${use_case}/radical.client.sandbox/.
cp -a /lustre/orion/scratch/hjjvd/chm136/test_nwchem /lustre/orion/scratch/hjjvd/chm136/${use_case}/.
cp -a ../postproduction /lustre/orion/scratch/hjjvd/chm136/${use_case}/.
cp -a *.png /lustre/orion/scratch/hjjvd/chm136/${use_case}/postproduction
cd /lustre/orion/scratch/hjjvd/chm136/
tar -zcf ${use_case}.tgz ${use_case}
