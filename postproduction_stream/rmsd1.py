import numpy as np
import adios2
import sys
import pandas as pd
import subprocess

fn = sys.argv[1]
compute_zcentroid = int(sys.argv[2])

fn_out = fn.replace("trajectory.bp", "rmsd.csv")

print(subprocess.getstatusoutput("hostname")[1])


with adios2.open(fn, "r") as fr:
    n = fr.steps()
    if n == 0:
        print("n = 0")
        sys.exit(0)

    rmsd = fr.read("rmsd", [], [], 0, n)
    gpstime = fr.read("gpstime", [], [], 0, n)
    if compute_zcentroid:
        zcentroid = fr.read("zcentroid", [], [], 0, n)

    print(
        f"{fn}: gps={gpstime[0][0]}, n={n}, min(rmsd) = {np.min(rmsd)}, max(rms) = {np.max(rmsd)}, mean(rmsd) = {np.mean(rmsd)}, median(rmsd) = {np.median(rmsd)}"
    )
    if compute_zcentroid:
        print(
            f"min(zcentroid) = {np.min(zcentroid)}, max(zcentroid) = {np.max(zcentroid)}, mean(zcentroid) = {np.mean(zcentroid)}, median(zcentroid) = {np.median(zcentroid)}"
        )

    if compute_zcentroid:
        pf = pd.DataFrame(
            {
                "gpstime": gpstime.reshape(
                    n,
                ),
                "rmsd": rmsd.reshape(
                    n,
                ),
                "zcentroid": zcentroid.reshape(
                    n,
                ),
            }
        )
    else:
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
