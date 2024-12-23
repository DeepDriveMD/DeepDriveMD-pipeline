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
from pathlib import Path
import sys

# the NWCHEM_TOP environment variable needs to be set to specify
# where the NWChem executable lives.
nwchem_top = None
deepmd_source_dir = None
test_data = Path("../../../../../data/h2co/system")
test_pdb = Path(test_data,"h2co-unfolded.pdb")
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
test_path = Path(sys.argv[1])
#os.mkdir(test_path)
os.chdir(test_path)
instance = sys.argv[2]
print (instance)
instance = Path(instance)
# print("Generate NWChem input files")
# inputs_cp = ase_nwchem.fetch_input(test_data)
# inputs_gn = ase_nwchem.perturb_mol(30,test_pdb)
# inputs = inputs_cp + inputs_gn
# print(inputs)
print("Run NWChem")
#for instance in inputs:
test_inp = instance.with_suffix(".nwi")
test_out = instance.with_suffix(".nwo")
ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
ase_nwchem.nwchem_is_successful(test_out)
print("Done NWChem")
# print("Extract NWChem results")
# test_dat = glob.glob("*.nwo")
# ase_nwchem.nwchem_to_raw(test_dat)
# print("Convert raw files to NumPy files")
# ase_nwchem.raw_to_deepmd(deepmd_source_dir)
# print("All done")
