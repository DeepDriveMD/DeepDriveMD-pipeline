#!/bin/bash
export TEST=/lustre/orion/scratch/hjjvd/chm136/test_nwchem
for ii in `seq 0 9`;
do
    for jj in `seq $((ii+1)) 9`;
    do
	echo "$ii $jj"
        sort -k 3 -n -r data_points_${ii}_${jj}.csv > data.csv
        ./plot_rmsd.py data.csv --image dp_${ii}_${jj}.png
    done
done
