import numpy as np
import adios2
import sys
import glob
import os

bpdir = sys.argv[1]
outdir = sys.argv[2]
highest_rmsd = float(sys.argv[3])

bpfiles = glob.glob(f"{bpdir}/*/agg.bp")
bpfiles.sort()

print("bpfiles = ", bpfiles)

for fn in bpfiles:
    taskid = int(os.path.basename(os.path.dirname(fn)).replace("task", ""))
    print(f"taskid = {taskid}")
    with adios2.open(fn, "r") as fr:
        n = fr.steps()
        rmsd = fr.read("rmsd", [], [], 0, n)
        positions = fr.read("positions", [0, 0], [504, 3], 0, n)

        highest = filter(lambda x: x[0] <= highest_rmsd, list(zip(rmsd, positions)))

        for r, p in highest:
            print("r = ", r)
            fn = f"positions_{taskid}_{r:.2f}.npy"
            np.save(outdir + "/" + fn, p)

        print("=" * 30)
        sys.stdout.flush()
