"""Test the ase_lammps functionality."""
import ase_lammps
import glob
import os
import subprocess
import sys
from deepdrivemd.sim.lammps.config import LAMMPSConfig
from pathlib import Path

cwd = os.getcwd()
train = Path(cwd,"../../models/deepmd")
pdb = Path(cwd,"../../../data/h2co/system/h2co-unfolded.pdb")
data_dir = "pdbs"
test_dir = "test_dir"
python_exec = sys.executable
freq = 10
os.mkdir(test_dir)
os.chdir(test_dir)

config = LAMMPSConfig()
config.output_path = test_dir
config.pdb_file = pdb
config.dump_yaml("config.yaml")
subprocess.run([python_exec,"../run_lammps.py","--config","config.yaml"])
