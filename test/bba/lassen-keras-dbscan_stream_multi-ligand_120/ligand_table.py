import glob
import os
import subprocess
import sys

import pandas as pd

target_dir = sys.argv[1]

ml_dir = "/usr/workspace/cv_ddmd/brace3/data/plpro_ddmd_input/"

dirs = glob.glob(ml_dir + "/*")
dirs.sort()

pdbs = []
tops = []
tdirs = []

for i in range(len(dirs)):
    print(i, dirs[i])
    sys.stdout.flush()
    pdb = os.path.basename(glob.glob(dirs[i] + "/*.pdb")[0])
    pdbs.append(pdb)
    top = os.path.basename(glob.glob(dirs[i] + "/*.prmtop")[0])
    tops.append(top)
    tdir = target_dir + "/" + pdb.replace(".pdb", "").replace("sys_", "")
    subprocess.getstatusoutput("mkdir -p " + tdir)
    subprocess.getstatusoutput(f"ln -sf {dirs[i]} {tdir}/system")
    tdirs.append(tdir)

df = pd.DataFrame({"dir": dirs, "pdb": pdbs, "top": tops, "tdir": tdirs})

df.to_csv(f"{target_dir}/ml_table.csv")
