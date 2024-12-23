#!/bin/bash
#SBATCH -A chm136
#SBATCH -J 1HP8_1_H
#SBATCH -o %x-%j.out
#SBATCH -t 8:00:00
#SBATCH -N 1
srun -N 1 -n 56 --tasks-per-node=56 /autofs/nccs-svm1_home1/hjjvd/DeepDriveMD/nwchem/bin/LINUX64/nwchem 1hp8_1_h.nw
