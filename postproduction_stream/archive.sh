#!/usr/bin/bash


tar cvf $1.tar $1/postproduction $1/molecular_dynamics_runs/stage0000/task*/*/embeddings $1/molecular_dynamics_runs/stage0000/task*/*/positions.npy $1/molecular_dynamics_runs/stage0000/task*/*/rmsd.csv $1/machine_learning_runs/stage0000/task0000/published_model/best.*


