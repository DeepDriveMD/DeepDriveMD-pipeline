import adios2
import numpy as np
from collections import Counter
import sys
import glob
import os

fns = glob.glob("../../Outputs/306/aggregation_runs/stage0000/task00*/agg.bp")
fns.sort()
fns.reverse()
# total_count = 19200
count = 200 * 12

for fn in fns:
    print("=" * 20 + ">")
    print(f"fn = {fn}")
    sys.stdout.flush()
    sim = int(os.path.basename(os.path.dirname(fn)).replace("task", ""))
    with adios2.open(fn, "r") as fh:
        total_count = fh.steps()
        iterations = int(total_count / count)
        print("total_count = ", total_count, " iterations = ", iterations)
        for i in range(iterations):
            print("=" * 10 + ">")
            positions = {}

            j = 0
            for fstep in fh:
                print("=" * 5 + ">")
                variables = fstep.available_variables()
                # print(variables)
                pshape = list(map(int, variables["positions"]["Shape"].split(",")))
                print(pshape)

                dir = int(fstep.read_string("dir")[0])
                step = fstep.read("step")

                print(
                    f"iteration i = {i}, j = {j}, sim = {dir}, agg = {sim}, step = {step}"
                )

                p = fstep.read("positions", [0, 0], pshape)
                print(f"p.shape = {p.shape}")

                try:
                    positions[(pshape[0], dir)].append(p)
                except Exception as e:
                    print(e)
                    positions[(pshape[0], dir)] = [p]

                j += 1
                print("=" * 5 + "<")
                sys.stdout.flush()
                if j == count:
                    break

            keys = list(positions.keys())
            keys.sort()
            print(f"keys={keys}")
            for k, dir in keys:
                print("=" * 5 + ">")
                print(f"k = {k}, dir = {dir}")
                fn1 = f"iteration_{i}_ligand_{k}_sim_{dir}_agg_{sim}.npy"
                print(f"fn1 = {fn1}")
                print(f"len(positions[(k, dir)]) = {len(positions[(k, dir)])}")
                c = Counter(list(map(lambda x: x.shape[0], positions[(k, dir)])))
                print("c = ", c)
                np.save(fn1, np.array(positions[(k, dir)]))
                print("=" * 5 + "<")

            print("=" * 10 + "<")
    print("=" * 20 + "<")
