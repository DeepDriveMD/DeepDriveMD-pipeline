import numpy as np
import adios2
import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD
import sys
import pandas as pd
from multiprocessing import Pool
import glob
import os

fn = sys.argv[1]
ps = int(sys.argv[2])
out = sys.argv[3]

def f(position):
    outlier_traj = mda.Universe(init_pdb, position)
    ref_traj = mda.Universe(ref_pdb_file)
    R = RMSD(outlier_traj, ref_traj, select = 'protein and name CA')
    R.run()
    return R.rmsd[:,2][0]

ref_pdb_file = '../../data/bba/ddmd_reference/1FME.pdb'

init_pdb = '../../data/bba/ddmd_input/1FME-0.pdb'

pf = pd.DataFrame(columns=["fstep", "step", "R"])

with adios2.open(fn, "r") as fr:
    n = fr.steps()
    vars = fr.available_variables()
    
    print(vars)
    
    name = 'positions'
    shape = list(map(int, vars[name]['Shape'].split(",")))
    zs = list(np.zeros(len(shape), dtype=np.int))
    positions = fr.read(name, zs, shape, 0, n)
    print(type(positions))
    print(positions.shape)
    sys.stdout.flush()
    
    name = 'step'
    steps = fr.read(name, [],[], 0, n)
    print(type(steps))
    print(steps.shape)
    print(steps)
    sys.stdout.flush()


with Pool(processes=ps) as pool:
    Rs = pool.map(f, positions)

pf['fstep'] = list(np.arange(len(steps)))
pf['step'] = steps
pf['R'] = Rs

# fn1 = os.path.basename(fn).replace(".bp",".csv")

pf.to_csv(out)
