"""Test the ase_lammps functionality."""
import ase_lammps
import glob
import os
from deepdrivemd.sim.lammps.config import LAMMPSConfig
from pathlib import Path

N2P2=1
DEEPMD=2
env_model = os.getenv("FF_MODEL")
if env_model == "DEEPMD":
    model = DEEPMD
elif env_model == "N2P2":
    model = N2P2
else:
    model = DEEPMD

cwd = os.getcwd()
pdb = Path(cwd,"../../../data/h2co/system/h2co-unfolded.pdb")
test_dir = "test_dir"
os.mkdir(test_dir)
config_file = Path(test_dir, "config.yaml")
lammpscfg = LAMMPSConfig(reference_pdb_file = pdb)
lammpscfg.dump_yaml(str(config_file))
cfg = LAMMPSConfig.from_yaml(str(config_file))
if model == DEEPMD:
    train = Path(cwd,"../../models/deepmd")
elif model == N2P2:
    train = Path(cwd,"../../models/n2p2")
data_dir = "pdbs"
trajectory = Path(cwd,test_dir,"trj_lammps.dcd")
if model == DEEPMD:
    freq = 100
    steps = 10000
elif model == N2P2:
    freq = 1
    steps = 100
os.chdir(test_dir)

ase_lammps.lammps_input(pdb,train,trajectory,freq,steps)
ase_lammps.run_lammps()
ase_lammps.lammps_get_devi(trajectory,pdb)
failed, struct = ase_lammps.lammps_questionable(0.1,0.3,freq)
if failed:
    print("Reject trajectory")
else:
    print("Accept trajectory")
print(struct)
trajectory = Path(cwd,test_dir,"trj_lammps.dcd")
hdf5_basename = Path(cwd,test_dir,"trj_lammps")
ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)
ase_lammps.lammps_contactmap(cfg,trajectory,pdb,hdf5_basename,freq,steps)

exit()

trajectories = glob.glob("scratch/trj_lammps*")
for trajectory in trajectories:
    print(trajectory)
    ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)

