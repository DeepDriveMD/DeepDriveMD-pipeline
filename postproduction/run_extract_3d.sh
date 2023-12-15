#!/bin/bash
export TEST=/lustre/orion/scratch/hjjvd/chm136/test_nwchem
for ii in `seq 0 9`;
do
    for jj in `seq $((ii+1)) 9`;
    do
        for kk in `seq $((jj+1)) 9`;
        do
            echo "$ii $jj $kk"
            ./extract_data.py \
                ${TEST}/machine_learning_runs/stage0009/task0000/stage0009_task0000.yaml \
                ${TEST}/machine_learning_runs/stage0009/task0000/checkpoint/epoch-50-20231211-185026.h5 \
                ${TEST}/molecular_dynamics_runs "$ii,$jj,$kk" data_points_${ii}_${jj}_${kk}.csv
        done
    done
done
