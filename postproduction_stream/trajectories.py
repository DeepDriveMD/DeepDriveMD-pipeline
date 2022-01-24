import adios2
import numpy as np
from collections import Counter
import sys
import glob
from pathlib import Path

dir_top = "/p/gpfs1/yakushin/Outputs"
sim = sys.argv[1]
dir_subtop = dir_top + "/" + sys.argv[1]
fns = glob.glob(dir_subtop + "/aggregation_runs/stage0000/task00*/agg.bp")
fns.sort()
dir_output = dir_subtop + "/postproduction/trajectories"
Path.mkdir(Path(dir_output), exist_ok=True)

start = 0
end = len(fns)

try:
    start = int(sys.argv[2])
    end = int(sys.argv[3])
except Exception as e:
    print(e)

print("start = ", start, " end = ", end)
sys.stdout.flush()


# fns.reverse()
# total_count = 19200
count = 200 * 12

for fn in fns[start:end]:
    with adios2.open(fn, "r") as fh:
        total_count = fh.steps()
        iterations = int(total_count / count)
        print("total_count = ", total_count, " iterations = ", iterations)
        for i in range(iterations - 1, 0, -1):
            start = count * i
            print("i = ", i)
            sys.stdout.flush()
            a = fh.read_string("dir", start, count)
            b = np.array(list(map(int, a)))

            s = fh.read("step", [], [], start, count)
            print("s.shape = ", s.shape)
            p = fh.read("positions", [0, 0], [504, 3], start, count)
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

                fn1 = f"{dir_output}/{i}_{cc}_step.npy"
                fn2 = f"{dir_output}/{i}_{cc}_positions.npy"

                # np.save(fn1, ss)
                np.save(fn2, pp)
