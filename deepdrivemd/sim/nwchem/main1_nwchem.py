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
import sys
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
test_data = Path("../../../../../data/h2co/system")
test_pdb = Path(test_data,"h2co-unfolded.pdb")
test_inp = "h2co.nwi"
test_out = "h2co.nwo"
test_path = Path("./test_dir")
curr_path = Path("./")
test_path = Path(sys.argv[1])
if not test_path.exists():
    os.makedirs(test_path,exist_ok=True)
os.chdir(test_path)
print("Generate NWChem input files")
inputs_path = Path(test_path,"inputs.txt")
if not inputs_path.exists():
    # We haven't run any DFT calculations yet so generate input files
    # and store the list of inputs files
    # - grab a bunch of predefined input files
    # - perturb the initial molecular structure to generate more inputs
    # Note that N2P2 gets hopelessly confused when the training set
    # contains a mix of chemical structures. So for N2P2 we stick to
    # just using perturbed structures of 1 chemical structure and 
    # nothing else.
    # For DeePMD a mixture of different chemical compounds is fine.
    # So in that case we can add single atoms, diatomics and other
    # small structures that are quick to evaluate and provide
    # additional information.
    inputs_cp = ase_nwchem.fetch_input(test_data)
    inputs_gn = ase_nwchem.perturb_mol(475,test_pdb)
    if model == DEEPMD:
        inputs = inputs_gn + inputs_cp
    elif model == N2P2:
        inputs = inputs_gn
    else:
        inputs = inputs_gn + inputs_cp
else:
    # We need to take new input files from the PDB structure generated
    # by the LAMMPS MD run
    pdbs_path = Path(sys.argv[2])
    inputs = []
    filename = Path(pdbs_path,"pdb_files.txt")
    tmp_path = Path("tmp.pdb")
    with open(str(filename), "r") as fp:
        lines = fp.readlines()
    for line in lines:
        pdb_path = Path(pdbs_path,line.strip())
        input_path = Path(test_path,Path(line.strip()).stem)
        input_name = input_path.with_suffix(".nwi")
        ase_nwchem.clean_pdb(pdb_path,tmp_path)
        ase_nwchem.nwchem_input(input_name,tmp_path)
        inputs.append(input_path)
with open("inputs.txt", "w") as f:
    for filename in inputs:
        print(str(filename), file=f)
print("Done NWChem input files")

