'''
A test driver for the code in nwchem.py

Basically this code makes sure that
1. We can generate a valid nwchemrc file
2. We can generate and run the input for the prepare stage
3. We can generate and run the input for the energy minimization
4. We can generate and run the input for the MD
5. We can generate and run the input for a subsequent MD 
6. We can generate and run the input for the trajectory conversion
In particular step 5 is important. We do not want to start from
scratch. Instead we want to start from an existing topology file
and restart file but generate an new trajectory file with 
additional time frames.
'''

import nwchem
import os
import subprocess
from pathlib import Path

# You'll have to set the NWCHEM_TOP environment variable to tell where
# NWChem lives. This location is installation dependent.
nwchem_top = None
test_pdb = "../../../../data/bba/system/1FME-unfolded.pdb"
test_path = Path("./test_dir")
curr_path = Path("./")
# Create directory for the test
os.mkdir(test_path)
os.chdir(test_path)
# Setting the system up
print("Set system up")
nwchem.make_nwchemrc(curr_path,nwchem_top)
print(" - Prepare the system")
nwchem.gen_input_prepare(test_pdb)
nwchem.run_nwchem(nwchem_top,"_prepare")
print(" - Minimize the energy")
nwchem.gen_input_minimize()
nwchem.run_nwchem(nwchem_top,"_minimize")
print(" - Replace restart file")
nwchem.replace_restart_file()
print(" - Run an equilibration simulation")
nwchem.gen_input_dynamics(False,0.002,0.004,310.15,0.2)
nwchem.run_nwchem(nwchem_top,"_equilibrate")
#subprocess.run(["cp","nwchemdat.out","equi.out"])
# Run a MD short simulation
print("Run a MD short simulation")
nwchem.gen_input_dynamics(True,0.002,0.000200,310.15,0.002)
nwchem.run_nwchem(nwchem_top,"_dynamics1")
#subprocess.run(["cp","nwchemdat.out","md_001.out"])
print("Run a MD short simulation")
nwchem.gen_input_dynamics(True,0.002,0.000200,310.15,0.002)
nwchem.run_nwchem(nwchem_top,"_dynamics2")
#subprocess.run(["cp","nwchemdat.out","md_002.out"])
# Convert the trajectory
print("Convert the trajectory")
nwchem.gen_input_analysis()
nwchem.run_nwchem(nwchem_top,"_analysis")
print("Read the trajectory data")
data = nwchem.read_trajectory("nwchemdat_md.xyz","nwchemdat_md.xyz")
print(data)


