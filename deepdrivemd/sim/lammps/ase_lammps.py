"""Define the setup for a short LAMMPS MD simulation using DeePMD

Note that DeepDriveMD mainly thinks in terms of biomolecular structures.
Therefore it passes molecular structures around in PDB files. LAMMPS is
a general MD code that is more commonly used for materials (DL_POLY is
another example of such an MD code). Hence PDB files are strangers in 
LAMMPS's midst, but with the DeePMD force field this should be workable.

Approach:
    - Take the geometry in a PDB file
    - Take the force specification
    - Take the temperature
    - Take the number of time steps
    - Write the input file
    - Run the MD simulation
    - Check whether any geometries were flagged
    - Convert the trajectory for DeepDriveMD
"""

import ase
import glob
import itertools
import MDAnalysis as mda
import deepdrivemd.models.n2p2.n2p2 as n2p2
import numpy as np
import operator
import os
import subprocess
import sys
from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_masses, chemical_symbols
from ase.io import iread
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.proteindatabank import read_proteindatabank, write_proteindatabank
from deepdrivemd.sim.lammps.config import LAMMPSConfig
from MDAnalysis.analysis import distances, rms
from mdtools.analysis.order_parameters import fraction_of_contacts
from mdtools.writers import write_contact_map, write_point_cloud, write_fraction_of_contacts, write_rmsd
from mdtools.nwchem.reporter import OfflineReporter
from os import PathLike
from pathlib import Path
from typing import List

N2P2   = 1
DEEPMD = 2
env_model = os.getenv('FF_MODEL')
if env_model == "DEEPMD":
    model = DEEPMD
elif env_model == "N2P2":
    model = N2P2
else:
    model = DEEPMD

class Atoms:
    """Atoms class to trick MD-tools reporter."""
    def __init__(self,atom_lst: List):
        self._atoms = atom_lst
    def atoms(self):
        return self._atoms

class Simulation:
    """Simulation class to trick MD-tools reporter.

    MD-tools (https://github.com/hjjvandam/MD-tools/tree/nwchem) was
    originally designed to support OpenMM calculations in DeepDriveMD.
    I just want to run LAMMPS, and use MD-tools to convert the DCD file
    produced to a contact map file in HDF5 as required by DeepDriveMD. 
    At the same time I don't want to bring all of OpenMM into my
    environment just to be able to run this conversion. This Simulation
    class with the Atoms class above allows me to give the OfflineReporter
    what it needs without bringing all the OpenMM baggage along.
    """
    def __init__(self,pdb_file):
        self.pdb_file = Path(pdb_file)
        universe = mda.Universe(pdb_file,pdb_file)
        atomgroup = universe.select_atoms("all")
        atoms = [ag for ag in atomgroup]
        self.topology = Atoms(atoms)

def _sort_uniq(sequence):
    """Return a sorted sequence of unique instances.

    See https://stackoverflow.com/questions/2931672/what-is-the-cleanest-way-to-do-a-sort-plus-uniq-on-a-python-list
    """
    return map(operator.itemgetter(0),itertools.groupby(sorted(sequence)))

def lammps_input(pdb: PathLike, train: PathLike, traj: PathLike, freq: int, steps: int) -> None:
    """Create the LAMMPS input file.

    The DeePMD models live in directories:

        {train}/train-*/compressed_model.pb

    The N2P2 models live in directories:

        {train}/train-*/weights.*.data

    The frequency specified here needs to match that in the
    trajectory checking. The trajectory will contain only 
    (#steps)/(freq) frames, whereas "model_devi.out" labels
    each structure by the original timestep ("model_devi.out" is
    produced only by the DeePMD force field, for N2P2 we need to
    generate something similar ourselves).

    Arguments:
    pdb -- the PDB file with the structure
    train -- the path to the directory above the DeePMD models
    traj -- the DCD trajectory file
    freq -- frequency of generating output
    """
    global model, DEEPMD, N2P2
    cwd = os.getcwd()
    temperature = 300.0
    subprocess.run(["cp",str(pdb),"lammps_input.pdb"])
    atoms = read_proteindatabank(pdb,index=0)
    pbc = atoms.get_pbc()
    if all(pbc):
        cell = atoms.get_cell()
    elif not any(pbc):
        cell = 2 * np.max(np.abs(atoms.get_positions())) * np.eye(3)
        atoms.set_cell(cell)
    lammps_data  = Path(cwd,"data_lammps_structure")
    lammps_input = Path(cwd,"in_lammps")
    lammps_trj   = Path(traj)
    lammps_out   = Path(cwd,"out_lammps")
    if model == DEEPMD:
        # Taking compressed models out for now due to disk space limitations.
        # The compressed models are 10x larger than the uncompressed ones
        # (also raising questions about what compression means here).
        #deep_models = glob.glob(str(Path(train,"train-*/compressed_model.pb")))
        deep_models = glob.glob(str(Path(train,"train-*/model.pb")))
    elif model == N2P2:
        deep_models = str(Path(train,"train-1"))
    else:
        raise RuntimeError(f"Illegal value for model {model}")
    with open(lammps_data,"w") as fp:
        write_lammps_data(fp,atoms)
    with open(lammps_input,"w") as fp:
        fp.write( "clear\n")
        fp.write( "atom_style   atomic\n")
        fp.write( "units        metal\n")
        fp.write( "atom_modify  sort 0 0.0\n\n")
        fp.write(f"read_data    {lammps_data}\n\n")
        if model == DEEPMD:
            pair_style = "pair_style   deepmd"
            for model in deep_models:
                pair_style += f" {model}"
            fp.write(f"{pair_style}\n")
            fp.write( "pair_coeff   * *\n\n")
        elif model == N2P2:
            pair_style = f"pair_style   nnp maxew 1000000 resetew yes dir {deep_models} emap \""
            fp.write(f"{pair_style}")
            element_list = list(enumerate(_sort_uniq(atoms.get_chemical_symbols())))
            for i, cs in element_list:
                ii = i+1
                fp.write(f"{ii}:{cs}")
                if ii < len(element_list):
                    fp.write(",")
            fp.write("\"\n")
            fp.write( "pair_coeff   * * 6.0\n\n")
        else:
            raise RuntimeError(f"Invalid value of model {model}")
        for i, cs in enumerate(_sort_uniq(atoms.get_chemical_symbols())):
            ii = i+1
            mass = atomic_masses[chemical_symbols.index(cs)]
            fp.write(f"mass {ii} {mass}\n")
        fp.write( "\n")
        fp.write( "fix      fix_nve all nve\n")
        fp.write( "minimize 1.0e-4 1.0e-6 100 1000\n")
        fp.write( "unfix    fix_nve\n")
        fp.write( "\n")
        fp.write( "timestep     0.000025\n") # was 0.001 
        fp.write(f"fix          fix_nvt  all nvt temp {temperature} {temperature} $(100.0*dt)\n")
        fp.write(f"dump         dump_all all dcd {freq} {lammps_trj}\n")
        fp.write( "thermo_style custom step temp etotal ke pe atoms\n")
        fp.write(f"thermo       {freq}\n")
        fp.write(f"run          {steps} upto\n")
        fp.write( "print        \"__end_of_ase_invoked_calculation__\"\n")
        fp.write(f"log          {lammps_out}\n")
            
def run_lammps() -> None:
    """Run a LAMMPS calculation.

    Note that ASE gets the LAMMPS executable from the
    environment variable ASE_LAMMPSRUN_COMMAND.
    """
    lammps_exe = Path(os.environ.get("ASE_LAMMPSRUN_COMMAND"))
    if not lammps_exe:
        raise RuntimeError("run_lammps: ASE_LAMMPSRUN_COMMAND undefined")
    if not Path(lammps_exe).is_file():
        raise RuntimeError("run_lammps: ASE_LAMMPSRUN_COMMAND("+str(lammps_exe)+") is not a file")
    with open("in_lammps","r") as fp_in:
        subprocess.run([lammps_exe],stdin=fp_in)

def lammps_get_devi(trj_file: PathLike, pdb_file: PathLike) -> None:
    """N2P2 does not produce "model_devi.out" so we need to create it

    DeePMD automatically compares the results from the different models
    for each of the structures in the trajectory, and summarizes the outcome
    in the file "model_devi.out". 

    N2P2 assumes that your model is fully trained for all conceivable cases
    before you start running any MD. So it doesn't have an internal measure 
    for the precision of the model. Hence we need to replicate this 
    feature.
    """
    if model == DEEPMD:
        return
    # make a sub-directory for each model
    scaling_name = Path("scaling.data")
    input_name   = Path("input.nn")
    train_path   = Path("..")/".."/".."/".."/"models"/"n2p2"
    scaling_path = train_path/"scaling"/"scaling.data"
    input_path   = train_path/"scaling"/"input.nn"
    if not input_path.exists():
        train_path   = Path("..")/".."/"n2p2"/"train-1"
        scaling_path = train_path/"scaling.data"
        input_path   = train_path/"input.nn"
        train_path   = Path("..")/".."/"n2p2"
    for ii in range(1,5):
        dir_name = f"model-{str(ii)}"
        os.makedirs(dir_name,exist_ok=True)
        os.chdir(dir_name)
        subprocess.run(["ln","-s",str(scaling_path),str(scaling_name)])
        subprocess.run(["ln","-s",str(input_path),  str(input_name)])
        weights = glob.glob(str(Path(train_path)/f"train-{str(ii)}"/"weights.???.data"))
        for pathname in weights:
            filename = os.path.basename(pathname)
            if not os.path.exists(filename):
                subprocess.run(["ln","-s",str(pathname),str(filename)])
        os.chdir("..")
    universe = mda.Universe(pdb_file,trj_file)
    selection = universe.select_atoms("all")
    with open("model_devi.out","w") as fdevi:
        step = -1
        fdevi.write("#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f\n")
        for ts in universe.trajectory:
            step += 1
            molecules = []
            for ii in range(1,5):
                dir_name = f"model-{str(ii)}"
                os.chdir(dir_name)
                write_input_data(selection)
                n2p2.run_predict()
                with open("output.data","r") as fp:
                    molecule = n2p2.read_molecule(fp)
                    molecules.append(molecule)
                os.chdir("..")
            (e_max,e_min,e_avg,f_max,f_min,f_avg) = n2p2.compare_molecules(molecules)
            fdevi.write(f" {step:11d}   {e_max:16e}   {e_min:16e}   {e_avg:16e}   {f_max:16e}   {f_min:16e}   {f_avg:16e}\n")

def write_input_data(selection) -> None:
    """Given a structure write it to input.data

    The input.data is the N2P2 structure file. This function writes 
    a single given structure as an input.data file. As we plan
    to give this file to nnp-predict we only need the atomic
    coordinates. The nnp-predict tool will produce the energy
    and forces for this structure.
    """
    with open("input.data","w") as fp:
        fp.write("begin\n")
        fp.write("comment structure\n")
        for atom in selection:
            x1, y1, z1    = atom.position
            e1            = atom.element
            fx1, fy1, fz1 = 0.0, 0.0, 0.0
            c1, n1        = 0.0, 0.0
            fp.write(f"atom {x1} {y1} {z1} {e1} {c1} {n1} {fx1} {fy1} {fz1}\n")
        fp.write("charge 0.0\n")
        fp.write("end\n")
    

def lammps_questionable(force_crit_lo: float, force_crit_hi: float, freq: int) -> List[int]:
    """Return a list of all structures with large force mismatches.

    There are two criteria. If the difference in the forces exceeds
    the lower criterion then the corresponding structure should be
    added to the training set. If the difference exceeds the higher
    criterion for any point then the errors are so severe that the
    trajectory should be considered non-physical. So its structures
    should not be used in the DeepDriveMD loop.

    Arguments:
    force_crit_lo -- the lower force criterion
    force_crit_hi -- the higher force criterion
    """
    structures = []
    failed = False
    with open("model_devi.out","r") as fp:
        # First line is just a line of headers
        line = fp.readline()
        # First line of real data
        line = fp.readline()
        while line:
            ln_list = line.split()
            struct_id = int(ln_list[0])
            error     = float(ln_list[4])
            if error > force_crit_lo:
                if struct_id % freq != 0:
                    raise RuntimeError("lammps_questionable: frequency mismatch")
                structures.append(int(struct_id/freq))
            if error > force_crit_hi:
                failed = True
            line = fp.readline()
    return (failed, structures)

def lammps_save_model_devi() -> None:
    """Copy model_devi.out to a unique name for future reference

    Model_devi.out details the deviation between the different DeePMD
    models. This information should be kept to evaluate how the DeePMD
    models improve over time with more training data. To generate
    a unique name we simply hash the contents of the file and append
    the hash to the filename.
    """
    import hashlib
    with open("model_devi.out","rb") as fp:
        lines = fp.readlines()
    h = hashlib.sha256()
    for line in lines:
        h.update(line)
    hashkey = h.hexdigest()
    with open("model_devi.out-"+str(hashkey),"wb") as fp:
        fp.writelines(lines)

#class lammps_txt_trajectory:
#    """A class to deal with LAMMPS trajectory data in txt format.
#
#    A class instance manages a single trajectory file.
#    - Creating an instance opens the trajectory file.
#    - Destroying an instance closes the trajectory file.
#    - Read will read the next timestep from the trajectory file.
#    """
#    def __init__(self, trj_file: PathLike, pdb_orig: PathLike):
#        """Create a trajectory instance for trj_file.
#
#        This constructor needs the PDB file from which the LAMMPS
#        calculation was generated. In generating the LAMPS input
#        the chemical element information was discarded. This means
#        that the Atoms objects contain chemical nonsense information.
#        By extracting the chemical element information from the 
#        PDB file this information can be restored before returning
#        the Atoms object.
#
#        Arguments:
#        trj_file -- the filename of the trajectory file
#        pdb_orig -- the filename of the PDB file 
#        """
#        self.trj_file = trj_file
#        self.trj_file_it = iread(trj_file,format="lammps-dump-text")
#        atoms = read_proteindatabank(pdb_orig)
#        self.trj_atomicno = atoms.get_atomic_numbers()
#        self.trj_symbols = atoms.get_chemical_symbols()
#
#    def next(self) -> ase.Atoms:
#        atoms = next(self.trj_file_it,None)
#        if atoms:
#            atoms.set_atomic_numbers(self.trj_atomicno)
#            atoms.set_chemical_symbols(self.trj_symbols)
#        return atoms

def lammps_to_pdb(trj_file: PathLike, pdb_file: PathLike, indeces: List[int], data_dir: PathLike):
    """Write timesteps from the LAMMPS DCD format trajectory to PDB files."""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not Path(data_dir).is_dir():
        raise RuntimeError(f"{data_dir} exists but is not a directory")
    hashno = str(abs(hash(trj_file)))
    universe = mda.Universe(pdb_file,trj_file)
    selection = universe.select_atoms("all")
    ii = 0
    istep_trj = -1
    pdb_list = []
    if ii >= len(indeces):
        # We are done
        with open("pdb_files.txt","w") as fp:
            for pdb_file in pdb_list:
                 print(pdb_file, file=fp)
        return
    istep_lst = indeces[ii] 
    for ts in universe.trajectory:
        istep_trj +=1
        while istep_lst <  istep_trj:
            ii += 1
            if ii >= len(indeces):
                # We are done
                with open("pdb_files.txt","w") as fp:
                    for pdb_file in pdb_list:
                         print(pdb_file, file=fp)
                return
            istep_lst = indeces[ii]
        print(f"lst, trj: {istep_lst} {istep_trj}")
        if istep_lst == istep_trj:
            # Convert this structure to PDB
            filename = Path(data_dir,f"atoms_{hashno}_{istep_trj}.pdb")
            with mda.Writer(filename,universe.trajectory.n_atoms) as wrt:
                wrt.write(selection)
            pdb_list.append(filename)
    with open("pdb_files.txt","w") as fp:
        for pdb_file in pdb_list:
            print(pdb_file, file=fp)

def lammps_contactmap(cfg: LAMMPSConfig, trj_file: PathLike, pdb_file: PathLike, hdf5_file: PathLike, report_steps: int, total_steps: int):
    """Write timesteps from the LAMMPS DCD format trajectory to PDB files."""
    hashno = str(abs(hash(trj_file)))
    trj = mda.Universe(pdb_file,trj_file)
    pdb = mda.Universe(pdb_file,pdb_file)
    sim = Simulation(pdb_file)
    frames_per_h5 = int(total_steps/report_steps)

    reporter = OfflineReporter(
                   hdf5_file,report_steps,frames_per_h5=frames_per_h5,
                   wrap_pdb_file=None,reference_pdb_file=pdb_file,
                   openmm_selection=cfg.lammps_selection,
                   mda_selection=cfg.mda_selection,
                   threshold=8.0,
                   contact_map=cfg.contact_map,
                   point_cloud=cfg.point_cloud,
                   fraction_of_contacts=cfg.fraction_of_contacts)
    num_frames = 0
    for ts in trj.trajectory:
        num_frames += 1
        reporter.report(sim,ts)
