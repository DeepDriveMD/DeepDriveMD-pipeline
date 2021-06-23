i=0
for bp in ../../Outputs/2/aggregation_runs/stage0000/*/agg.bp; do
    echo "i = $i"
    echo "bp = $bp"
    python adios2pandasR4.py $bp 40 $i.csv > $i.out 2>$i.err
    i=$((i+1))
done

