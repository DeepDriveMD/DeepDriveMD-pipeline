'''
Define the set up to run a single point NWChem Gradient simulation

In general this should be simple:
    - Take a geometry
    - Take a basis set specification and a density functional
    - Write the input file
    - Run the DFT calculation
    - Analyze the results
'''
# We are using the Atomic Simulation Environment (ASE)
# because ASE has been used to generate the files DeePMD-kit
# needs to train by the DeePMD-kit developers.
import ase 
import os
import string
import subprocess
from os import PathLike
from pathlib import Path
from ase.calculators.nwchem import NWChem
from ase.io.nwchem import write_nwchem_in, read_nwchem_out

# From https://www.weizmann.ac.il/oc/martin/tools/hartree.html [accessed March 28, 2024]
hartree_to_ev = 27.211399

def nwchem_input(inpf: PathLike, pdb: PathLike) -> None:
    '''
    Generate an NWChem input file

    Take the input structure (a PDB file) and create an ASE NWChem
    calculator for a DFT gradient calculation. The calculator is
    is returned. Because of the way ASE is designed we actually return
    the molecule object with the calculator attached.

    We use the SCAN functional because of https://doi.org/10.1021/acs.jctc.2c00953.
    We use unrestricted DFT because at transition states you may not 
    have a closed shell electron density.
    We use a TZVP basis set because that is the smallest basis set for which 
    reasonable results can be expected.
    '''
    molecule = ase.io.read(pdb)
    fp = open(inpf,"w")
    opts = dict(basis="cc-pvtz",
                dft=dict(xc="scan",
                         mult=1,
                         odft=None,
                         direct=None,
                         maxiter=500,
                         noprint="\"final vectors analysis\""),
                theory="dft")
    # ASE is going to insist on this directory for the 
    # permanent_dir and scratch_dir so we have to make 
    # sure it exists
    if not os.path.exists("./nwchem"):
        os.mkdir("./nwchem")
    elif not os.path.isdir("./nwchem"):
        raise OSError("./nwchem exists but is not a directory")
                
    write_nwchem_in(fp,molecule,["forces"],True,**opts)
    fp.close()

def run_nwchem(nwchem_top: PathLike, inpf: PathLike, outf: PathLike) -> None:
    '''
    Run the NWChem calculation

    NWChem is invoked with fixed command line parameters.
    - nwchem.nwi - input file
    - nwchem.nwo - output file
    The executable name is constructed from the NWCHEM_TOP
    environment variable as NWCHEM_TOP/bin/LINUX64/nwchem.
    In principle LINUX64 could be different for different
    operating systems, but at present LINUX64 is correct for
    almost any computer system (this used to be very
    different).

    The environment variable NWCHEM_TASK_MANAGER specifies
    the task manager to start the parallel calculation.
    Common choices would be "srun", "mpirun", or "mpiexec"

    The environment variable NWCHEM_NPROC specifies the
    number of MPI processes to run NWChem on.
    '''
    if not nwchem_top:
        nwchem_top = os.environ.get("NWCHEM_TOP")
    if nwchem_top:
        nwchem_exe = nwchem_top+"/bin/LINUX64/nwchem"
    else:
        raise RuntimeError("run_nwchem: NWCHEM_TOP undefined")
    if not Path(nwchem_exe).is_file():
        raise RuntimeError("run_nwchem: NWCHEM_EXE("+nwchem_exe+") is not a file")
    nwchem_task_man = os.environ.get("NWCHEM_TASK_MANAGER")
    if not nwchem_task_man:
        nwchem_task_man = "mpirun"
    nwchem_nproc = os.environ.get("NWCHEM_NPROC")
    if not nwchem_nproc:
        nwchem_nproc = os.environ.get("SLURM_NTASKS")
    if not nwchem_nproc:
        nwchem_nproc = os.environ.get("PBS_NP")
    if not nwchem_nproc:
        #nwchem_nproc = "16"
        nwchem_nproc = "1"
    fpout = open(outf,"w")
    subprocess.run([nwchem_task_man,"-np",nwchem_nproc,nwchem_exe,inpf],stdout=fpout,stderr=subprocess.STDOUT)
    fpout.close()

def _make_atom_list(symbols: list,atomicnos: list) -> list:
    '''
    Turn the list of chemical symbols and atomic numbers into a list of tuples

    In order the present the data correctly to DeePMD we need to
    sort the atoms. To facilitate this we create a list of tuples
    where each tuple consists of:

       (index, symbol, atomic number)

    where index is the atom position in the original list, 
    symbol is the chemical symbol of the atom, and atomic number
    is the corresponding atomic number of the element.
    '''
    result = []
    len_symbols = len(symbols)
    len_atomicno = len(atomicnos)
    if not len_symbols == len_atomicno:
        raise RuntimeError("List of chemical symbols and atomic numbers are "+
                           "of different length "+str(len_symbols)+" "+str(len_atomicno))
    for ii in range(len_symbols):
        result.append((ii,symbols[ii],atomicnos[ii]))
    return result

def _make_molecule_name(tuples: list) -> str:
    '''
    DeePMD needs a kind of bruto formula for the molecule as a name

    The way DeePMD stores its training data we need separate directories
    for every different "bruto formula" in the training set. 

    To generate this name we need to count how often every element appears 
    in the atom list. Then we need to string the chemical symbols with their
    counts together in a string to generate the name.

    Note that for formaldehyde this function will produce h2c1o1. While this
    will likely annoy chemists we have to it this way otherwise you cannot
    distinguish between different molecules like C Au and Ca U, now these
    would produce c1au1 and ca1u1 which clearly are different (essentially
    we use the count BOTH to report the count AND as a separator between
    elements).
    '''
    # There are 118 chemical elements but the atomic numbers are base 1 instead of base 0
    symbols = [""] * 119
    counts = [0] * 119
    for atm_tuple in tuples:
        index, symbol, atomicno = atm_tuple
        symbols[atomicno] = symbol.lower()
        counts[atomicno] += 1
    result = ""
    for ii in range(119):
        if counts[ii] > 0:
            result += symbols[ii] + str(counts[ii])
    return result

def _write_type_map(fp: PathLike) -> None:
    '''
    Write the "standard" type map to the file provided
    '''
    with open(fp,"w") as mfile:
        for ii in range(1,118):
            mfile.write(ase.data.chemical_symbols[ii].lower()+" ")
        mfile.write(ase.data.chemical_symbols[118].lower())

def _write_type(fp: PathLike, tuples: list) -> None:
    '''
    Write DeePMD's type file

    The type file contains a single line with the atomic number minus 1 
    for each atom in the molecule
    '''
    with open(fp,"w") as mfile:
        for atm_tuple in tuples:
            index, symbol, atomicno = atm_tuple
            atomicno -= 1
            mfile.write(str(atomicno)+" ")

def _write_energy(fp: PathLike, energy: float) -> None:
    '''
    Append the energy in eV to the energy file
    '''
    with open(fp,"a") as mfile:
        mfile.write(str(energy*hartree_to_ev)+"\n")

def _write_atmxyz(fp: PathLike, xyz: list, atmtuples: list, convert: float) -> None:
    '''
    Add a line with atomic x,y,z quantities to quantity file

    Because coordinates and forces are all 3D quantities we can use the
    same rountine to write either.

    This function is a little bit more involved because:
    - the have to be sorted according to the data in type.raw
    - xyz is a list of lists where for every atom you have a list of x,y,z
    Assumptions:
    - atmtuples is sorted on the atomic numbers
    - coordinates are provide in Angstrom
    - forces are provided in Hartree/Angstrom
    - convert is the appropriate conversion factor for DeePMD
    '''
    with open(fp,"a") as mfile:
        for tup in atmtuples:
            index, symbol, atomicno = tup
            xx, yy, zz = xyz[index]
            xx *= convert
            yy *= convert
            zz *= convert
            mfile.write(f'{xx} {yy} {zz} ')
        mfile.write("\n")

def nwchem_to_deepmd(nwofs: list) -> None:
    '''
    Extract data from NWChem outputs and store them in a form suitable for DeePMD

    DeePMD uses a batched learning approach. I.e. the training data is split into
    batches, and the training loops over batches to update the moded weights and
    biases. For all data points in a batch the chemistry needs to be the same, 
    meaning that every data point must have:

    - the same number of atoms
    - the same numbers of atoms of each chemical element
    - the same ordering of the atoms.

    For a given batch there are a number of files:

    - type_map.raw - translates atom types to chemical elements (a single line)
    - type.raw     - lists the atom types in a molecular structure in 0 based
                     type numbers (a single line)
    - coord.raw    - lists the atom positions of all atoms per line
    - force.raw    - lists the atomic forces of all atoms per line
    - energy.raw   - lists the total energy per line

    For finite systems (i.e. no periodic boundary conditions) there needs 
    to be a file with the name "nopbc" in the data directory.

    The coord.raw, force.raw, and energy.raw files should be converted into
    NumPy files. The type.raw and type_map.raw files are used as plain text.
    The coordinate, force, and energy files may be split into batches.
    Overall this gives us a data organization like:

    - mol_a/
      - type.raw
      - type_map.raw
      - set.000/
        - coord.npy
        - energy.npy
        - force.npy
      - set.001/
        - coord.npy
        - energy.npy
        - force.npy
    - mol_b/
      - type.raw
      - type_map.raw
      - set.000/
        - coord.npy
        - energy.npy
        - force.npy

    Obviously there is a lot of uncertainty about how this data is used. I.e.
    can I just use arbitary type data, for example set the atom type to be
    the atom number minus 1, or is this data being used in some fancy way?
    For example Uranyl UO2, can I just set the atom types to be 91 7 7,
    or should I compress this list to 0 1 1. In the former case I could simply
    keep the type map constant and just list all elements from the periodic 
    table, with the benefit that all atom types are defined the same way
    for all conceivable molecules. Or is someone going to use the atom type
    as an array index and specifying atom types 91 7 7 is going to create 
    some huge table? Who knows?

    Given all the uncertainties the following approach is selected (for now).
    The type_map.raw file simply contains all elements of the periodic table,
    as a results the types in type.raw simply consist of the atomic numbers
    minus 1. At worst this will generate data structures that are 100 times
    larger than they need to be. Because the neural networks in DeePMD are just
    a few kB a piece this will waste a most a few MB of memory, which seems
    acceptable. 

    Molecules are canonicalized by sorting the structures on the atomic
    numbers of the elements. The molecule names are constructed by
    concatenating the chemical symbol and the correspond atom count
    in the structure for all the constituent elements.
    '''
    for nwof in nwofs:
        fp = open(nwof,"r")
        data = read_nwchem_out(fp,slice(-1,None,None))
        fp.close()
        atoms = data[0]
        calc = atoms.get_calculator()
        # NWChem DFT energy in Hartree
        energy = calc.get_potential_energy()
        # Chemical symbols of the atoms
        symbols = atoms.get_chemical_symbols()
        # Atomic numbers of the atoms
        atomicno = atoms.get_atomic_numbers()
        # NWChem atomic positions in Angstrom
        positions = atoms.get_positions()
        # NWChem atomic forces in Hartree/Angstrom
        forces = calc.get_forces()
        atom_list = _make_atom_list(symbols,atomicno)
        atom_list.sort(key=lambda tup: tup[2])
        mol_name = Path("mol_" + _make_molecule_name(atom_list))
        if not mol_name.exists():
            os.mkdir(mol_name)
            fp = mol_name/"type_map.raw"
            _write_type_map(fp)
            fp = mol_name/"type.raw"
            _write_type(fp,atom_list)
            fp = open(mol_name/"nopbc","w")
            fp.close()
        elif not mol_name.isdir():
            raise OSError(mol_name+" exists but is not a directory")
        _write_energy(mol_name/"energy.raw",energy)
        _write_atmxyz(mol_name/"coord.raw", positions, atom_list, 1.0)
        _write_atmxyz(mol_name/"force.raw", forces, atom_list, hartree_to_ev)
