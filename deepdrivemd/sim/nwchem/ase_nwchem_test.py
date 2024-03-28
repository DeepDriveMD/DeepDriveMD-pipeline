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

# the NWCHEM_TOP environment variable needs to be set to specify
# where the NWChem executable lives.
nwchem_top = None
test_pdb = "../../../../data/h2co/system/h2co-unfolded.pdb"
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
os.mkdir(test_path)
os.chdir(test_path)
print("Generate NWChem input file")
ase_nwchem.nwchem_input(test_inp,test_pdb)
print("Run NWChem")
ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
print("Extract NWChem results")
test_dat = glob.glob("*.nwo")
ase_nwchem.nwchem_to_deepmd(test_dat)
print("All done")
