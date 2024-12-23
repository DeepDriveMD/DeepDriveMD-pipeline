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
deepmd_source_dir = None
test_pdbs = Path("../../lammps/test_dir/pdbs")
test_path = Path("./test_dir")
curr_path = Path("./")
os.chdir(test_path)
print("Generate NWChem input files")
inputs = ase_nwchem.gen_new_inputs(test_pdbs)
print(inputs)
print("Run NWChem")
new_list = []
for instance in inputs:
    test_inp = instance.with_suffix(".nwi")
    test_out = instance.with_suffix(".nwo")
    ase_nwchem.run_nwchem(nwchem_top,test_inp,test_out)
    new_list.append(test_out)
print("Extract NWChem results")
ase_nwchem.nwchem_to_raw(new_list)
print("Convert raw files to NumPy files")
ase_nwchem.raw_to_deepmd(deepmd_source_dir)
print("All done")
