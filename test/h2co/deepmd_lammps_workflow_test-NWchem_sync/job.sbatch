#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --account "csihpc"
#SBATCH --partition "csi"
#SBATCH --time 24:00:00

## ENVIRONMENT
#export CWD=`pwd`
export RADICAL_PROFILE="TRUE"
#export RADICAL_BASE="$CWD/radical.pilot.sandbox"
export RADICAL_PILOT_BASE="/hpcgpfs01/work/csi/hvandam/DeepDriveMD-pipeline/test/h2co/deepmd_lammps_workflow_test-NWchem_sync"
#export RADICAL_BASE="/hpcgpfs01/work/csi/hvandam/DeepDriveMD-pipeline/test/h2co/deepmd_lammps_workflow_test-NWchem_sync"
export RADICAL_SMT="1"
export MKL_THREADING_LAYER=INTEL
export MKL_SERVICE_FORCE_INTEL=1

id -un
id -Gn

## EXEC
. /hpcgpfs01/work/csi/hvandam/pydeepmd-3.11/bin/activate
pwd
hostname
module list
nvidia-smi
status=$?
echo "nvida-smi status: $status"
echo "ldd lmp_mpi"
ldd /hpcgpfs01/work/csi/hvandam/DeePMD/GPU/lammps/src/lmp_mpi
python ../../../deepdrivemd/NWchem_sync.py -c ./config.yaml
#python ../../../deepdrivemd/NWchem_Adapt.py -c ./config.yaml
#/bin/bash "-l" "/scratch/hvandam/radical.pilot.sandbox/re.session.ic2submit01.sdcc.bnl.gov.hvandam.019836.0003/pilot.0000//bootstrap_0.sh" "-p" "pilot.0000" "-s" "re.session.ic2submit01.sdcc.bnl.gov.hvandam.019836.0003" "-m" "local" "-r" "local" "-b" "anaconda" "-v" "/sdcc/u/hvandam/.conda/envs/pydeepmd" "-y" "180" "-z" "re.session.ic2submit01.sdcc.bnl.gov.hvandam.019836.0003.pmgr_launching.0000.tgz"
