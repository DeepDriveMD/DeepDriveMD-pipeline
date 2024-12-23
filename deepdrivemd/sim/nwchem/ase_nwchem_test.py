'''
A test driver for the code in ase_nwchem.py

These tests make sure that
1. We can generate a valid NWChem input file with ASE
2. We can run an NWChem calculation for the energy and gradient
3. We can use ASE to extract the results of NWChem
4. We can store the results in a format suitable for DeePMD
'''

import os
import ase_nwchem
import glob
import n2p2
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

# the NWCHEM_TOP environment variable needs to be set to specify
# where the NWChem executable lives.
nwchem_top = None
deepmd_source_dir = None
test_data = Path("../../../../data/h2co/system")
test_pdb = Path(test_data,"h2co-unfolded.pdb")
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
os.mkdir(test_path)
os.chdir(test_path)
print("Generate NWChem input files")
inputs_cp = ase_nwchem.fetch_input(test_data)
inputs_gn = ase_nwchem.perturb_mol(400,test_pdb)
if model == DEEPMD:
    inputs = inputs_gn + inputs_cp
elif model == N2P2:
    inputs = inputs_gn
else:
    inputs = inputs_gn + inputs_cp
print(inputs)
print("Run NWChem")
for instance in inputs:
    test_inp = instance.with_suffix(".nwi")
    test_out = instance.with_suffix(".nwo")
    ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
print("Extract NWChem results")
test_dat = glob.glob("*.nwo")
ase_nwchem.nwchem_to_raw(test_dat)
if model == DEEPMD:
    print("Convert raw files to NumPy files")
    ase_nwchem.raw_to_deepmd(deepmd_source_dir)
print("Convert raw files to N2P2 files")
n2p2.generate_n2p2_test_files_for_all_folders()
print("All done")
