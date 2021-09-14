import adios2
import numpy as np
from collections import Counter
import sys
import glob

fns = glob.glob("../../Outputs/301/aggregation_runs/stage0000/task00*/agg.bp")
# fns.reverse()
# total_count = 19200
count = 200 * 12

for fn in fns:
    with adios2.open(fn, "r") as fh:
        total_count = fh.steps()
        iterations = int(total_count / count)
        print("total_count = ", total_count, " iterations = ", iterations)
        for i in range(iterations):
            start = count * i
            print("i = ", i)
            sys.stdout.flush()
            a = fh.read_string("dir", start, count)
            b = np.array(list(map(int, a)))

            s = fh.read("step", [], [], start, count)
            print("s.shape = ", s.shape)
            p = fh.read("positions", [0, 0], [171898, 3], start, count)
            print("p.shape = ", p.shape)

            c = Counter(b)
            print("c = ", c)
            for cc in c.keys():
                print("cc = ", cc)
                sys.stdout.flush()
                selection = b == cc
                print("selection = ", selection)
                ss = s[selection]
                print("ss.shape = ", ss.shape)
                pp = p[selection]
                print("pp.shape = ", pp.shape)

                fn1 = f"{i}_{cc}_step.npy"
                fn2 = f"{i}_{cc}_positions.npy"

                np.save(fn1, ss)
                np.save(fn2, pp)