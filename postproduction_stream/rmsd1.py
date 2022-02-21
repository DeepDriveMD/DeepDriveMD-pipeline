import numpy as np
import adios2
import sys
import pandas as pd
import subprocess

fn = sys.argv[1]
fn_out = fn.replace("trajectory.bp", "rmsd.csv")

print(subprocess.getstatusoutput("hostname")[1])


with adios2.open(fn, "r") as fr:
    n = fr.steps()
    if n == 0:
        print("n = 0")
        sys.exit(0)

    rmsd = fr.read("rmsd", [], [], 0, n)
    gpstime = fr.read("gpstime", [], [], 0, n)
    print(
        f"{fn}: gps={gpstime[0][0]}, n={n}, min(rmsd) = {np.min(rmsd)}, max(rms) = {np.max(rmsd)}, mean(rmsd) = {np.mean(rmsd)}, median(rmsd) = {np.median(rmsd)}"
    )

    pf = pd.DataFrame(
        {
            "gpstime": gpstime.reshape(
                n,
            ),
            "rmsd": rmsd.reshape(
                n,
            ),
        }
    )
    pf.to_csv(fn_out, index=False)
    sys.stdout.flush()
