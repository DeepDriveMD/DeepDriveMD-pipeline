"""Test the ase_lammps functionality."""
import ase_lammps
import glob
import os
from pathlib import Path
from deepdrivemd.sim.lammps.config import LAMMPSConfig
import sys

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
test_dir = Path(sys.argv[1])
config_file = Path(test_dir, "config.yaml")
cfg = LAMMPSConfig.from_yaml(str(config_file))
pdb = Path(sys.argv[2],"*.pdb")
if len(sys.argv) == 3:
    train = Path(test_dir,"../../../deepmd")
else:
    train = Path(sys.argv[3])
pdbs = glob.glob(str(pdb))
pdb = Path(pdbs[0])
data_dir = "pdbs"
trajectory = Path(cwd,test_dir,"trj_lammps.dcd")
hdf5_basename = Path(cwd,test_dir,"trj_lammps")
print("Begin LAMMPS run")
print(str(sys.argv),file=sys.stderr)
if model == DEEPMD:
    freq = 100
    steps = 10000
else:
    freq = 1
    steps = 100
if not test_dir.exists():
    os.makedirs(test_dir,exist_ok=True)
os.chdir(test_dir)

ase_lammps.lammps_input(pdb,train,trajectory,freq,steps)
ase_lammps.run_lammps()
ase_lammps.lammps_get_devi(trajectory,pdb)
failed, struct = ase_lammps.lammps_questionable(0.0003,0.3,freq)
success = not failed
with open("lammps_success.txt", "w") as fp:
    print(success, file=fp)
if failed:
    print("Reject trajectory")
else:
    print("Accept trajectory")
print(struct)
ase_lammps.lammps_to_pdb(trajectory,pdb,struct,data_dir)
ase_lammps.lammps_contactmap(cfg,trajectory,pdb,hdf5_basename,freq,steps)
ase_lammps.lammps_save_model_devi()
print("Done  LAMMPS run")
