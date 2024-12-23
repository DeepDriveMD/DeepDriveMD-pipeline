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
import glob
import numpy
import os
import random
import shutil
import string
import subprocess
from os import PathLike
from pathlib import Path, PurePath
from typing import List, Tuple
from ase.atoms import Atoms
from ase.calculators.nwchem import NWChem
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io.nwchem import write_nwchem_in, read_nwchem_out
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank

# From https://www.weizmann.ac.il/oc/martin/tools/hartree.html [accessed March 28, 2024]
hartree_to_ev = 27.211399

N2P2=1
DEEPMD=2
env_model = os.getenv("FF_MODEL")
if env_model == "DEEPMD":
    model = DEEPMD
elif env_model == "N2P2":
    model = N2P2
else:
    model = DEEPMD

def perturb_mol(number: int, pdb: PathLike) -> List[PathLike]:
    """Write input files for a number of molecular structures.

    To initialize a collection of structures for DeePMD to train on we need 
    to generate such a collection. A simple way to do this is to take the
    initial structure we have been given and subject it to a random walk.
    Each resulting structure is stored until we have the prescribed number
    of structures. 
    In practice this is a bad idea as a random walk can blunder into completely
    unrealistic parts of conformational space. So instead we now take the
    initial structure every time and perturb it. We also make sure that the
    initial structure itself is included in the training set.

    The names of the new structures will be derived from the input PDB filename
    and include a number to ensure uniqueness. A list of structure names will
    be returned. As a side effect input files for every structure will be produced.

    From the list returned input file names can be constructed by adding ".nwi"
    and output file name by adding ".nwo"

    Arguments:
    number -- the number of perturbed structure to generate
    pdb -- the PDB file with the initial molecular structure
    """
    # Create a random number generator instance. 
    # We need to pass this to rattle, otherwise rattle 
    # internally creates and seeds its own random
    # number generator. Because this freezes the
    # random number sequence every rattle call always does
    # the exact same thing (not very random at all).
    global model, DEEPMD, N2P2
    random = numpy.random.default_rng()
    with open(pdb,"r") as fp:
        atoms = read_proteindatabank(pdb,index=0)
    symbols = atoms.get_chemical_symbols()
    atomicno = atoms.get_atomic_numbers()
    atom_list = _make_atom_list(symbols,atomicno)
    atom_list.sort(key=lambda tup: tup[2])
    mol_name = _make_molecule_name(atom_list)
    name_list = []
    for ii in range(number):
         with open(pdb,"r") as fp:
             atoms = read_proteindatabank(pdb,index=0)
         # Perturb the atom positions
         #
         # For N2P2 we need a bunch of structures closely around
         # the reference structure. The references structure will be
         # used as the starting point for any MD. If this structure
         # is poorly represented then the code will throw "extrapolation"
         # warnings and either aborts or generate abhorrently bad
         # trajectories.
         #
         # Note that N2P2 cannot deal with unbound atoms. As the bonding
         # cutoff is typically 6 and bondlength are typically < 2 then
         # as long as we don't perturb atom positions by more than 2
         # bound atoms will still be bound after applying the perturbation.
         if ii == 0:
             pass
         elif ii < 25:
             atoms = rattle(atoms=atoms,limit=0.1,rng=random)
         elif ii < 75:
             atoms = rattle(atoms=atoms,limit=0.25,rng=random)
         elif ii < 175:
             atoms = rattle(atoms=atoms,limit=0.5,rng=random)
         elif ii < 375:
             atoms = rattle(atoms=atoms,limit=1.0,rng=random)
         else:
             #atoms.rattle(stdev=0.005,rng=random)
             atoms = rattle(atoms=atoms,limit=2.0,rng=random)
         tmpfile = Path("./tmp.pdb")
         with open(tmpfile,"w") as fp:
             write_proteindatabank(fp,atoms)
         # Add the name to the return list
         fname = Path(f"{mol_name}_{ii:06d}")
         name_list.append(fname)
         # Add the extension for the input file and write the input
         inpname = fname.with_suffix(".nwi")
         nwchem_input(inpname,tmpfile)
    return name_list

def rattle(atoms: Atoms, limit: float = 1.0, seed: int = None, rng = None) -> Atoms:
    """My rattle implementation

    ASE's rattle draws perturbations from a normal distribution. N2P2 will 
    terminate the training if a structure contains an atom without neighbors.
    So for N2P2 we need perturbations that have strict limits on the 
    displacement, which is incompatible with normal distributions. 
    Therefore, we need our own implementation of rattle that draws
    perturbations from a uniform distribution.

    DeePMD did not have any problems with isolated atoms, but it was very
    problematic to install on HPC facilities. So we had to make a choice
    and we chose to switch to N2P2, because it is easier for me to rewrite
    code than to fix arcane installation conflicts.

    - atoms is the ASE Atoms structure
    - limit is the limit on the range of perturbations, i.e. the range is [-limit,limit]
    """
    if seed is not None and rng is not None:
        raise ValueError('Please do not provide both seed and rng.')

    if rng is None:
        if seed is None:
            seed = 42
        rng = np.random.RandomState(seed)
    positions = atoms.arrays['positions']
    atoms.set_positions(positions +
                        rng.uniform(low=-limit,high=limit, size=positions.shape))
    return atoms
    
def clean_pdb(pdb: PathLike,tmp: PathLike) -> None:
    """Supress ASE junk.

    ASE inherently assumes that any PDB file it writes is a crystal structure.
    The CRYST1 line it adds reinforces that and when ASE writes an NWChem 
    input file it will write the geometry as if you are using the planewave
    code. This breaks everything when you try to do Gaussian basis set 
    calculations. So we need to remove the CRYST1 lines. 

    Another nasty thing is that ASE adds ENDMDL lines. This means that the PDB
    files it writes ends as:

       ENDMDL
       END

    This causes another problem. The ENDMDL line completes the molecular structure
    and starts a new one. As the next line is END this last molecular structure
    is an empty one. The protein reader read_proteindatabank by default returns
    the last structure from the PDB file. I.e. if ASE wrote the PDB file you get
    an empty structure! Fortunately, you can just ask the structure corresponding
    to index=0 and get what you need. So for now I am not going to worry about
    this, but this is definitely something to keep in mind.
    """
    with open(pdb,"r") as fp_in:
        with open(tmp,"w") as fp_out:
            line = fp_in.readline()
            while line:
                if not line[0:6] == "CRYST1":
                    fp_out.write(line)
                line = fp_in.readline()

def gen_new_inputs(pdb_path: PathLike) -> List[PathLike]:
    """Write input files for a number of molecular structures.

    From, for example, a prior molecular dynamics run we get an additional
    set of PDB files. This function converts those PDB into new input
    files and returns the list of those input files. We need to check
    that the new input filenames are unique so we don't overwrite 
    anything.

    From the list returned input file names can be constructed by adding ".nwi"
    and output file name by adding ".nwo"

    Arguments:
    pdb_path -- the path where the new PDB files reside
    """
    pdbs = glob.glob(str(Path(pdb_path,"*.pdb")))
    name_list = []
    ii = 0
    for pdb in pdbs:
        tmp_pdb = "tmp.pdb"
        clean_pdb(pdb,tmp_pdb)
        with open(tmp_pdb,"r") as fp:
            atoms = read_proteindatabank(fp,index=0)
        symbols = atoms.get_chemical_symbols()
        atomicno = atoms.get_atomic_numbers()
        atom_list = _make_atom_list(symbols,atomicno)
        atom_list.sort(key=lambda tup: tup[2])
        mol_name = _make_molecule_name(atom_list)
        fname = Path(f"{mol_name}_{ii:06d}")
        while fname.with_suffix(".nwi").exists():
            ii += 1
            fname = Path(f"{mol_name}_{ii:06d}")
        name_list.append(fname)
        # Add the extension for the input file and write the input
        inpname = fname.with_suffix(".nwi")
        nwchem_input(inpname,tmp_pdb)
    return name_list

def fetch_input(data: PathLike) -> List[PathLike]:
    """Copy pre-existing NWChem input files."""
    path = Path(data,"*.nwi")
    src_inputs = glob.glob(str(path))
    name_list = []
    for inp in src_inputs:
        name = Path(inp).name
        stem = Path(inp).stem
        subprocess.run(["cp",str(inp),str(name)])
        name_list.append(Path(stem))
    return name_list

def nwchem_input(inpf: PathLike, pdb: PathLike) -> None:
    """Generate an NWChem input file.

    Take the input structure (a PDB file) and create an ASE NWChem
    calculator for a DFT gradient calculation. The calculator is
    is returned. Because of the way ASE is designed we actually return
    the molecule object with the calculator attached.

    We use the SCAN functional because of https://doi.org/10.1021/acs.jctc.2c00953.
    We use unrestricted DFT because at transition states you may not 
    have a closed shell electron density.
    We use a TZVP basis set because that is the smallest basis set for which 
    reasonable results can be expected.

    Arguments:
    inpf -- the name of the input file to be generated
    pdb -- the PDB file containing the chemical structure
    """
    with open(pdb,"r") as fp:
        molecule = read_proteindatabank(fp,index=0)
    name = Path(inpf).name
    name = str(name).replace(".nwi","_dat")
    fp = open(inpf,"w")
    opts = dict(label=name,
                basis="cc-pvdz",
                symmetry="c1",
                dft=dict(xc="scan",
                         mult=1,
                         cgmin=None,
                         direct=None,
                         maxiter=150, # for testing
                         mulliken=None,
                         noprint="\"final vectors analysis\""),
                theory="dft")
                
    write_nwchem_in(fp,molecule,["forces"],True,**opts)
    fp.close()

def run_nwchem(nwchem_top: PathLike, inpf: PathLike, outf: PathLike) -> None:
    """Run the NWChem calculation

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

    Arguments:
    nwchem_top -- specification of NWCHEM_TOP as an argument
    inpf -- the name of the NWChem input file
    outf -- the name of the NWChem output file
    """
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
    name = Path(inpf).name
    name = str(name).replace(".nwi","_dat")
    # ASE is going to insist on this directory for the 
    # permanent_dir and scratch_dir so we have to make 
    # sure it exists
    newpath = "./"+name
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    elif not os.path.isdir(newpath):
        raise OSError(f"{newpath} exists but is not a directory")
    fpout = open(outf,"w")
    #FIXME We need to pick the right task manager so we can start calculation on the right resource
    #DEBUG now we just run plain NWChem without mpirun/srun/etc.
    #subprocess.run(["/hpcgpfs01/ic2software/openmpi/4.1.5-gcc-12.3.0/bin/ompi_info"],stdout=fpout,stderr=subprocess.STDOUT)
    #subprocess.run([nwchem_task_man,"-n",nwchem_nproc,"--cpus-per-task=1","--ntasks-per-core=1",nwchem_exe,inpf],stdout=fpout,stderr=subprocess.STDOUT)
    subprocess.run([nwchem_exe,inpf],stdout=fpout,stderr=subprocess.STDOUT)
    fpout.close()
    # Now cleanup the data files
    subprocess.run(["rm","-rf",newpath])

def _make_atom_list(symbols: List[str],atomicnos: List[int]) -> List[Tuple[int,str,int]]:
    """Turn the list of chemical symbols and atomic numbers into a list of tuples.

    In order to present the data correctly to DeePMD we need to
    sort the atoms. To facilitate this we create a list of tuples
    where each tuple consists of:

       (index, symbol, atomic number)

    where index is the atom position in the original list, 
    symbol is the chemical symbol of the atom, and atomic number
    is the corresponding atomic number of the element.

    Arguments:
    symbols -- the list of chemical symbols of the atoms in the molecular structure
    atomicno -- the list of atomic numbers of the atoms in the molecular structure
    """
    result = []
    len_symbols = len(symbols)
    len_atomicno = len(atomicnos)
    if not len_symbols == len_atomicno:
        raise RuntimeError("List of chemical symbols and atomic numbers are "+
                           "of different length "+str(len_symbols)+" "+str(len_atomicno))
    for ii in range(len_symbols):
        result.append((ii,symbols[ii],atomicnos[ii]))
    return result

def _make_molecule_name(tuples: List[Tuple[int,str,int]]) -> str:
    """Return a kind of bruto formula for the molecule as a name.

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

    Arguments:
    tuples -- list of tuples (index, symbol, atomic number)
    """
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

def _write_type_map(fp: PathLike, tuples: List[Tuple[int,str,int]]) -> None:
    """Write the type map to file.
    
    Arguments:
    fp -- the filename of the type map file
    tuples -- list of tuples (index, symbol, atomic number) sorted by atomic number
    """
    with open(fp,"w") as mfile:
        old_atomicno = -1
        for atm_tuple in tuples:
            index, symbol, atomicno = atm_tuple
            if atomicno != old_atomicno:
                old_atomicno = atomicno
                mfile.write(ase.data.chemical_symbols[atomicno].lower()+" ")

def _write_type(fp: PathLike, tuples: List[Tuple[int,str,int]]) -> None:
    """Write DeePMD's type file

    The type file contains a single line with the atomic number minus 1 
    for each atom in the molecule

    Arguments:
    fp -- the filename of the types file
    tuples -- list of tuples (index, symbol, atomic number) sorted by atomic number
    """
    with open(fp,"w") as mfile:
        old_atomicno = -1
        atomtype = -1
        for atm_tuple in tuples:
            index, symbol, atomicno = atm_tuple
            if atomicno != old_atomicno:
                old_atomicno = atomicno
                atomtype += 1
            mfile.write(str(atomtype)+" ")

def _write_energy(fp: PathLike, energy: float) -> None:
    """Append the energy in eV to the energy file.

    Arguments:
    fp -- the file name for the energy.raw file
    energy -- the energy provide by ASE in eV
    """
    with open(fp,"a") as mfile:
        mfile.write(str(energy)+"\n")

def _write_atmxyz(fp: PathLike, xyz: List[List[float]], atmtuples: List[Tuple[int,str,int]], convert: float) -> None:
    """Add a line with atomic x,y,z quantities to quantity file.

    Because coordinates and forces are all 3D quantities we can use the
    same rountine to write either.

    This function is a little bit more involved because:
    - the coordinates have to be sorted according to the data in type.raw
    - xyz is a list of lists where for every atom you have a list of x,y,z
    Assumptions:
    - atmtuples is sorted on the atomic numbers
    - coordinates are provide in Angstrom
    - forces are provided in eV/Angstrom
    - convert is the appropriate conversion factor for DeePMD

    Arguments:
    fp -- the file name
    xyz -- list of coordinates in the original atom ordering
    atmtuples -- list of tuples (index, symbol, atomic number) sorted by atomic number
    convert -- the conversion factor for NWChem to DeePMD
    """
    with open(fp,"a") as mfile:
        for tup in atmtuples:
            index, symbol, atomicno = tup
            xx, yy, zz = xyz[index]
            xx *= convert
            yy *= convert
            zz *= convert
            mfile.write(f'{xx} {yy} {zz} ')
        mfile.write("\n")

def _write_box(fp,box=None) -> None:
    """Append the simulation box to box.raw.

    If box is None then we don't have periodic boundary conditions and
    we write a default 1x1x1 box.

    If box is not None then we write the three lattice vectors to box.raw.

    Arguments:
    fp -- the file name
    box -- the lattice vectors
    """
    with open(fp,"a") as mfile:
        if box:
            for tup in box:
                xx, yy, zz = tup
                mfile.write(f'{xx} {yy} {zz} ')
            mfile.write("\n")
        else:
            mfile.write("1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0\n")

class split_tvt:
    """A class to help with splitting data sets into training, validation, or testing sets."""
    splits = [0.8,0.9,1.0] # corresponds to 80% training, 10% validation, and 10% testing
    def __init__(self,splits: List[float]=None):
        """Constructor to allow setting the splits.

        The splits if given specify the proportions of the three categories.
        The default setting corresponds to 80% training, 10% validation, 10% testing.
        The given splits will be normalized and converted to make the selection
        easy. See function training_or_validate_or_test for details on selection.

        Arguments:
        splits -- a list of 3 non-negative numbers
        """
        if splits:
            if not len(splits) == 3:
                raise RuntimeError("The list of splits must contain 3 elements not "+str(len(splits)))
            total = 0.0
            for rr in splits:
                 total += rr
                 if rr < 0.0:
                     raise RuntimeError("The splits must be non-negative")
            if total <= 0.0:
                raise RuntimeError("The sum of splits must be positive")
            self.splits[0] = splits[0]/total
            self.splits[1] = (splits[0]+splits[1])/total
            self.splits[2] = 1.0

    def training_or_validate_or_test(self) -> str:
        """Select whether the current instance is part of the training, validation or test set.

        Based on a random number in the range [0,1] a selection is made for
        the current instance. Based on the selection a string is returned that can be one of

        - "training" : an element of the training set
        - "validate" : an element of the validation set
        - "testing"  : an element of the test set

        Note that it is assumed that entropy from the system is used to initialize the 
        pseudo random number generator. Hence no explicit seed is used. This is significant
        because the Python code is expected to be launched many times in the workflow. Using
        a fixed seed would generate the same sequence every time.
        """
        rnd = random.uniform(0.0,1.0)
        if rnd <= self.splits[0]:
            return "training"
        elif rnd <= self.splits[1]:
            return "validate"
        elif rnd <= self.splits[2]:
            return "testing"
        else:
            raise RuntimeError(f"Should not get here! {rnd} not <= {self.splits[2]}?")

def _global_chemical_symbols(mols: List[PathLike]) -> List[str]:
    """Return the set of chemical symbols from all the type_map.raw files.

    The list returned is sorted in alphabetical order.

    Arguments:
    mols -- the list of all directories containing training data
    """
    global_sym = []
    for mol in mols:
       path = Path(mol,"type_map.raw")
       with open(path,"r") as fp:
           line = fp.readline()
       elements = line.split()
       global_sym += elements
    # Use set to remove duplicates
    tmp = set(global_sym)
    # Turn unique elements back into a sorted list
    global_sym = sorted([x for x in tmp])
    return global_sym

def _remap_types(mols: List[PathLike], global_sym: List[str]) -> None:
    """Replace type_map.raw and type.raw with ones based on the global chemical symbols list."""
    for mol in mols:
        path_type_map = Path(mol,"type_map.raw")
        path_type     = Path(mol,"type.raw")
        with open(path_type_map,"r") as fp:
            line = fp.readline()
        chem_symb = line.split()
        with open(path_type,"r") as fp:
            line = fp.readline()
        chem_type = [int(x) for x in line.split()]
        chem_type = [global_sym.index(chem_symb[x]) for x in chem_type]
        with open(path_type_map,"w") as fp:
            for x in global_sym:
                fp.write(f"{x} ")
        with open(path_type,"w") as fp:
            for x in chem_type:
                y = str(x)
                fp.write(f"{y} ")

def _harmonize_atom_types() -> None:
    """Harmonize the type_map.raw and type.raw files across the entire training set."""
    # Gather all training data directories
    mols = glob.glob("**/*_mol_*",recursive=True)
    # Gather all chemical elements from the type_map.raw files
    global_symbols = _global_chemical_symbols(mols)
    # Replace the type_map.raw and type.raw files
    _remap_types(mols,global_symbols)

def nwchem_to_raw(nwofs: List[PathLike]) -> None:
    """Extract data from NWChem outputs and store them in a raw form suitable for DeePMD

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
    a few kB a piece this will waste at most a few MB of memory, which seems
    acceptable. 

    Molecules are canonicalized by sorting the structures on the atomic
    numbers of the elements. The molecule names are constructed by
    concatenating the chemical symbol and the correspond atom count
    in the structure for all the constituent elements.

    This function generates the ".raw" files, see the
    raw_to_deepmd function for the conversion to the final NumPy files.

    Comment on the suggestions given above: First of all, ASE tends to sort
    chemical elements alphabetically. So listing all the elements in the
    order of increasing atomic number causes mismatches between the trained
    models and the atom types in the molecular dynamics simulations. Hence
    we are forced to use alphabetic ordering everywhere. Second, the initial
    memory impact assessment turned out to be way off the mark. Providing
    one type_map.raw instance listing the entire periodic table of the elements
    caused TensorFlow to run out of memory. Hence the type_map.raw files should
    include a more limited set of elements. Third, whereas the chemical element
    information is provided to DeePMD during the training stage, LAMMPS that is 
    used for the molecular dynamics simulations only has atom type numbers. So
    we have to expect that the truly used atom information is in the type.raw
    files. This means that we have to harmonize the type_map.raw and type.raw
    files across the entire training data set used to train a particular model.

    Arguments:
    nwofs -- a list of NWChem output files
    """
    splitter = split_tvt([90.0,10.0,0.0])
    for nwof in nwofs:
        try:
            with open(nwof,"r") as fp:
                data = read_nwchem_out(fp,slice(-1,None,None))
            atoms = data[0]
            calc = atoms.get_calculator()
            # NWChem DFT energy in eV
            energy = calc.get_potential_energy()
            # Chemical symbols of the atoms
            symbols = atoms.get_chemical_symbols()
            # Atomic numbers of the atoms
            atomicno = atoms.get_atomic_numbers()
            # NWChem atomic positions in Angstrom
            positions = atoms.get_positions()
            # NWChem atomic forces in eV/Angstrom
            forces = calc.get_forces()
        except (PropertyNotImplementedError, ValueError):
            # If the geometry has atoms too close together NWChem
            # will detect this and refuse to run a calculation on
            # an unphysical (which will most fail badly if it was
            # run due to numerical issues stemming from massive
            # linear dependencies in the basis set).
            #
            # If the DFT calculation did not converge then ASE
            # will raise a PropertyNotImplementedError exception.
            #
            # In these cases we should move the output file (if the
            # calculation did not converge in 500 iterations then it
            # is clearly not sensible anyway), and skip to the next
            # output file.
            new_path = Path(nwof).with_suffix(".failed")
            os.replace(nwof,new_path)
            continue
        except (OSError):
            continue
        atom_list = _make_atom_list(symbols,atomicno)
        atom_list.sort(key=lambda tup: tup[1])
        if len(atom_list) <= 1:
            # Single atom chemical systems should always be added to the training set.
            # These systems do not make sense in the validation set:
            # - They have only a single geometry
            # - The information they provide is unique
            # - The information they provide is uniquely important as it pins the
            #   energy of a particular atom down. Given that DeePMD writes the
            #   total energy as a sum of atomic energies knowing what the individual
            #   energies are seems key.
            data_set = "training"
        else:
            data_set = splitter.training_or_validate_or_test()
        mol_name = Path(data_set + "_mol_" + _make_molecule_name(atom_list))
        if not mol_name.exists():
            os.mkdir(mol_name)
            fp = mol_name/"type_map.raw"
            _write_type_map(fp,atom_list)
            fp = mol_name/"type.raw"
            _write_type(fp,atom_list)
            fp = open(mol_name/"nopbc","w")
            fp.close()
        elif not mol_name.is_dir():
            raise OSError(mol_name+" exists but is not a directory")
        _write_energy(mol_name/"energy.raw",energy)
        _write_atmxyz(mol_name/"coord.raw", positions, atom_list, 1.0)
        _write_atmxyz(mol_name/"force.raw", forces, atom_list, 1.0)
        _write_box(mol_name/"box.raw")
    _harmonize_atom_types()

def nwchem_is_successful(nwof: PathLike) -> None:
    """Check whether an NWChem calculation ran successfully

    If not successful rename the output to *.failed.

    Arguments:
    nwof -- an NWChem output files
    """
    try:
        with open(nwof,"r") as fp:
            data = read_nwchem_out(fp,slice(-1,None,None))
        atoms = data[0]
        calc = atoms.get_calculator()
        # NWChem DFT energy in eV
        energy = calc.get_potential_energy()
        # Chemical symbols of the atoms
        symbols = atoms.get_chemical_symbols()
        # Atomic numbers of the atoms
        atomicno = atoms.get_atomic_numbers()
        # NWChem atomic positions in Angstrom
        positions = atoms.get_positions()
        # NWChem atomic forces in eV/Angstrom
        forces = calc.get_forces()
    except (PropertyNotImplementedError, ValueError):
        # If the geometry has atoms too close together NWChem
        # will detect this and refuse to run a calculation on
        # an unphysical (which will most fail badly if it was
        # run due to numerical issues stemming from massive
        # linear dependencies in the basis set).
        #
        # If the DFT calculation did not converge then ASE
        # will raise a PropertyNotImplementedError exception.
        #
        # In these cases we should move the output file (if the
        # calculation did not converge in 500 iterations then it
        # is clearly not sensible anyway), and skip to the next
        # output file.
        new_path = Path(nwof).with_suffix(".failed")
        os.replace(nwof,new_path)

def raw_to_deepmd(deepmd_source_dir: PathLike) -> None:
    """Convert collections of ".raw" files into the training batches DeePMD expects

    DeePMD provides a script to convert the ".raw" data files into NumPy files
    that it uses. We'll just use that script. This script needs to be run in 
    a directory corresponding to one particular molecular structure.
    In principle it should be possible to train on multiple molecular structures.
    So we'll just loop over all molecular directories and run the script in
    each.

    The DeePMD conversion script is called "raw_to_set.sh". It is supposed to
    to live at "$deepmd_source_dir/data/raw/raw_to_set.sh". The location
    of deepmd_source_dir can be passed in as an argument, alternatively 
    we'll check the environment variables, and the system path.

    Arguments:
    deepmd_source_dir -- the path to the DeePMD source code directory
    """
    if not deepmd_source_dir:
        deepmd_source_dir = Path(os.environ.get("deepmd_source_dir"))
    if deepmd_source_dir:
        raw_to_set = Path(deepmd_source_dir,"data/raw/raw_to_set.sh")
    else:
        raw_to_set = Path(shutil.which('raw_to_set.sh'))
    if not raw_to_set:
        raise RuntimeError("raw_to_set.sh not found! Set deepmd_source_dir environment variable or put raw_to_set.sh in your PATH!")
    if not raw_to_set.is_file():
        raise RuntimeError(raw_to_set+" is not a file! Set deepmd_source_dir environment variable or put raw_to_set.sh in your PATH!")
    if not os.access(raw_to_set, os.X_OK):
        raise RuntimeError(raw_to_set+" is not executable!")
    mols = glob.glob("**/*_mol_*",recursive=True)
    cwd = Path(os.getcwd())
    for moldir in mols:
        moldir = Path(moldir)
        os.chdir(moldir)
        subprocess.run([raw_to_set])
        os.chdir(cwd)
