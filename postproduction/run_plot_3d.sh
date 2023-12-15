#!/bin/bash
export TEST=/lustre/orion/scratch/hjjvd/chm136/test_nwchem
for ii in `seq 0 9`;
do
    for jj in `seq $((ii+1)) 9`;
    do
        for kk in `seq $((jj+1)) 9`;
        do
            echo "$ii $jj $kk"
            sort -k 4 -n -r data_points_${ii}_${jj}_${kk}.csv > data.csv
            ./plot_rmsd.py data.csv --image dp_${ii}_${jj}_${kk}.png
        done
    done
done
