import glob
import itertools
import math
import numpy as np
import operator
import os
from pathlib import Path
import deepdrivemd.models.n2p2.sfparamgen as sfp
import subprocess
import sys
import typing

class Molecule:
    '''A simple class to store molecules

    This class consists of a number of fields:
    - chemical_symbols: a list of chemical symbols for the atoms
    - coordinates: a list of atomic positions, each position is a list of x, y, and z
    - forces: a list of atomic forces, each force is a list of f_x, f_y, and f_z
    - energy: the energy associated with the molecule
    '''
    chemical_symbols = None
    coordinates = None
    forces = None
    energy = None
    def __init__(self,symbols=None,positions=None):
        '''Initialize a molecule instance
        '''
        self.num_forces = 0
        self.num_positions = 0
        self.num_symbols = 0
        if not symbols is None:
            self.num_symbols = len(symbols)
            self.chemical_symbols = []
            for symbol in symbols:
                self.chemical_symbols.append(symbol)
        if not positions is None:
            self.num_positions = len(positions)
            self.coordinates = []
            for coord in positions:
                if len(coord) != 3:
                    raise RuntimeError(f"Atom has {str(len(coord))} coordinates instead of 3: {str(coord)}")
                self.coordinates.append(coord)
        if self.num_symbols != 0:
            if self.num_positions != 0:
                if self.num_symbols != self.num_position:
                    raise RuntimeError(f"There should be equal numbers of chemical symbols {str(num_symbols)} and atomic coordinates {str(num_positions)}")

    def set_chemical_symbols(self,symbols):
        '''Set the chemical symbols for the atoms
        '''
        if self.num_positions != 0:
            if len(symbols) != self.num_positions:
                raise RuntimeError(f"The number of symbols {str(len(symbols))} should match the number of atoms")
        self.num_symbols = len(symbols)
        self.chemical_symbols = []
        for symbol in symbols:
            self.chemical_symbols.append(symbol)

    def set_positions(self,positions):
        '''Set the positions for the atoms
        '''
        if self.num_symbols != 0:
            if len(positions) != self.num_symbols:
                raise RuntimeError(f"The number of positions {str(len(positions))} should match the number of atoms")
        self.num_positions = len(positions)
        self.positions = []
        for position in positions:
            self.positions.append(position)

    def set_forces(self,forces):
        '''Set the forces for the atoms
        '''
        if self.num_symbols != 0:
            if len(forces) != self.num_symbols:
                raise RuntimeError(f"The number of forces {str(len(forces))} should match the number of atoms")
        self.num_forces = len(forces)
        self.forces = []
        for force in forces:
            self.forces.append(force)

    def set_energy(self,energy):
        '''Set the energy for the molecule
        '''
        self.energy = energy

    def get_energy(self):
        return self.energy

    def get_forces(self):
        return self.forces

    def get_positions(self):
        return self.positions

    def get_chemical_symbols(self):
        return self.chemical_symbols

def read_molecule(fp):
    '''Read a molecule from an input.data file

    We assume that the file is open and the file object
    is provided in fp. Next a molecule instance is created,
    the relevant pieces of information are scanned from the file
    and added to the molecule object, which is returned.

    The input data format follows the pattern

        begin
        comment anything goes here
        atom x y z s c n fx fy fz
        atom ...
        energy e
        charge 0.0
        end

    where

        x, y, z    - the atomic coordinates
        s          - the chemical symbol
        c          - the partial charge (not used)
        n          - the atomic number (not used)
        fx, fy, fz - the atomic forces
        e          - the molecular energy

    This implementation scans the file for the "begin" keyword and
    then reads and parses the lines until the "end" keyword is encountered.
    Upon return the file object will be positioned just beyond the "end"
    line.
    '''
    molecule = Molecule()
    energy = None
    positions = []
    forces = []
    symbols = []
    line = fp.readline()
    while not line.startswith("begin"):
        line = fp.readline()
    line = fp.readline()
    while not line.startswith("end"):
        if line.startswith("begin"):
            raise RuntimeError("Invalid file format. New \"begin\" present before closing \"end\".")
        elif line.startswith("comment"):
            pass
        elif line.startswith("atom"):
            components = line.split()
            x = float(components[1])
            y = float(components[2])
            z = float(components[3])
            s = str(components[4])
            c = float(components[5])
            n = float(components[6])
            fx = float(components[7])
            fy = float(components[8])
            fz = float(components[9])
            symbols.append(s)
            positions.append([x,y,z])
            forces.append([fx,fy,fz])
        elif line.startswith("energy"):
            components = line.split()
            energy = float(components[1])
        elif line.startswith("charge"):
            pass
        else:
            raise RuntimeError(f"Invalid keyword: {line}")
        line = fp.readline()
    molecule.set_energy(energy)
    molecule.set_chemical_symbols(symbols)
    molecule.set_positions(positions)
    molecule.set_forces(forces)
    return molecule

def write_molecule(fp,molecule):
    '''Write the molecule to file in the input.data format

    The file format in explained in the read_molecule
    function.
    '''
    energy = molecule.get_energy()
    symbols = molecule.get_chemical_symbols()
    positions = molecule.get_positions()
    forces = molecule.get_forces()
    fp.write("begin\n")
    for ii in range(len(symbols)):
        s = symbols[ii]
        x, y, z = positions[ii]
        fx, fy, fz = forces[ii]
        c, n = 0.0, 0.0
        fp.write(f"atom {x} {y} {z} {s} {c} {n} {fx} {fy} {fz}\n")
    fp.write("energy {energy}\n")
    fp.write("end\n")

def compare_vectors(veca,vecb):
    '''Compare two vector in 3D space

    Returns the length of the difference vector
    '''
    dx = veca[0] - vecb[0]
    dy = veca[1] - vecb[1]
    dz = veca[2] - vecb[2]
    r  = math.sqrt(dx*dx+dy*dy+dz*dz)
    return r

def compare_molecule_pair(mola: Molecule, molb: Molecule) -> (float, float, float, float, float, float):
    '''Compare a single pair of molecules

    See also the discussion in the compare_molecules function.
    '''
    energy_a = mola.get_energy()
    energy_b = molb.get_energy()
    e_max_diff = abs(energy_a - energy_b)
    e_min_diff = e_max_diff
    e_avg_diff = e_max_diff
    forces_a = mola.get_forces()
    forces_b = molb.get_forces()
    lena = len(forces_a)
    lenb = len(forces_b)
    if not lena == lenb:
        raise RuntimeError("molecule have different numbers of atoms")
    f_max_diff = 0.0
    f_min_diff = sys.float_info.max
    f_avg_diff = 0.0
    for ii in range(lena):
        r = compare_vectors(forces_a[ii],forces_b[ii])
        f_max_diff = max(f_max_diff,r)
        f_min_diff = min(f_min_diff,r)
        f_avg_diff = f_avg_diff + r/lena
    return (e_max_diff,e_min_diff,e_avg_diff,f_max_diff,f_min_diff,f_avg_diff)

def compare_molecules(molecules: list[Molecule]) -> (float, float, float, float, float, float):
    '''Compare the molecule data of multiple molecules

    We compare all pairs of molecules in the list of molecules provide.
    We compares energies and atomic forces. We calculate the maximum, 
    minimum and average absolute difference.

    Note that to compare the forces in an coordinate invariant way 
    we need to calculate the difference of the forces for an atom,
    and then compute the length of the difference vector.
    '''
    lenm = len(molecules)
    fac = 2.0/((lenm-1)*lenm)
    e_max_diff = 0.0
    e_min_diff = sys.float_info.max
    e_avg_diff = 0.0
    f_max_diff = 0.0
    f_min_diff = sys.float_info.max
    f_avg_diff = 0.0
    for ii in range(lenm):
        for jj in range(ii):
            (e_max,e_min,e_avg,f_max,f_min,f_avg) = compare_molecule_pair(molecules[ii],molecules[jj])
            e_max_diff = max(e_max_diff,e_max)
            e_min_diff = min(e_min_diff,e_min)
            e_avg_diff = e_avg_diff + e_avg*fac
            f_max_diff = max(f_max_diff,f_max)
            f_min_diff = min(f_min_diff,f_min)
            f_avg_diff = f_avg_diff + f_avg*fac
    return (e_max_diff,e_min_diff,e_avg_diff,f_max_diff,f_min_diff,f_avg_diff)


def read_elements(fname: Path) -> (list[str], list[int]):
    '''Read the elements in the training set

    The chemical elements are list in the comment line in the input.data
    file. The elements are the string in round brackets. This string
    is extracted, converted into a list and returned. For example
    a comment line will look like

        comment h4c2o1 (C H O)

    We also count how many atoms there are of each element and return
    these counts in a second list. We need these counts to screen 
    the symmetry functions. E.g. if there is only 1 Oxygen atom you
    cannot have an O-O pair, if you only have 2 Carbons you cannot
    have a C-C-C bond angle, etc.
    '''
    with open(fname,'r') as fp:
        while True:
            entry = fp.readline()
            if entry.startswith("comment"):
                #
                tmp1 = entry.split("(")[1]
                tmp2 = tmp1.split(")")[0]
                elements = tmp2.split()
                #
                tmp1 = entry.split()[1]
                # For each element find the corresponding count
                counts = []
                for element in elements:
                    index_el = tmp1.index(element.lower())
                    index_other = sys.maxsize
                    for other in elements:
                        index = tmp1.index(other.lower())
                        if index > index_el:
                            if index < index_other:
                                index_other = index
                    if index_other == sys.maxsize:
                        # element is the last element in the string
                        count = int(tmp1[index_el+len(element):])
                    else:
                        # index_el and index_other bracket the count
                        count = int(tmp1[index_el+len(element):index_other])
                    counts.append(count)
                return (elements, counts)

def gen_symfunc(elements: list[str], fname: Path, r_cutoff: float = 6.0, counts: list[int] = None) -> None:
    '''Generate the symmetry functions

    The N2P2 approach needs symmetry functions as inputs to the NNP
    setup. The functions are generated based on:
    - the chemical elements involved
    - the design rules preferred
    The symmetry functions are written out to one of the N2P2
    input files. The input file name is given in fname and this function
    appends the symmetry functions to that file.
    There are two sets of rules derived from two papers:
    - 'gastegger2018' from https://doi.org/10.1063/1.5019667
    - 'imbalzano2018' from https://doi.org/10.1063/1.5024611
    '''
    gen = sfp.SymFuncParamGenerator(elements,r_cutoff)
    rule = 'gastegger2018'
    rule = 'imbalzano2018'
    mode = 'center'
    mode = 'shift'
    r_lower = 0.01
    r_upper = r_cutoff
    if rule == 'imbalzano2018':
        r_lower = None
        r_upper = None
    nb_param_pairs=5
    gen.generate_radial_params(rule=rule,mode=mode,nb_param_pairs=nb_param_pairs)

    with open(fname,'a') as fp:
        gen.symfunc_type = 'radial'
        if not counts is None:
            gen.filter_element_combinations(counts)
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'weighted_radial'
        if not counts is None:
            gen.filter_element_combinations(counts)
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.zetas = [1.0,6.0]
        gen.symfunc_type = 'angular_narrow'
        if not counts is None:
            gen.filter_element_combinations(counts)
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'angular_wide'
        if not counts is None:
            gen.filter_element_combinations(counts)
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)
        gen.symfunc_type = 'weighted_angular'
        if not counts is None:
            gen.filter_element_combinations(counts)
        gen.write_settings_overview(fileobj=fp)
        gen.write_parameter_strings(fileobj=fp)

def run_scaling():
     '''Run nnp-scaling

     One needs to run nnp-scaling to compute symmetry function
     statistics that the training will use. 
     '''
     n2p2_root = os.environ.get("N2P2_ROOT")
     if not n2p2_root:
         scaling_exe = "nnp-scaling"
     else:
         scaling_exe = Path(n2p2_root) / "bin" / "nnp-scaling"
     scaling_exe = str(scaling_exe)
     nnp_nproc = 1
     with open("nnp-scaling.out","w") as fpout:
         subprocess.run([scaling_exe,"100"],stdout=fpout,stderr=subprocess.STDOUT)

def run_training():
     '''Run nnp-training
     '''
     n2p2_root = os.environ.get("N2P2_ROOT")
     if not n2p2_root:
         training_exe = "nnp-train"
     else:
         training_exe = Path(n2p2_root) / "bin" / "nnp-train"
     training_exe = str(training_exe)
     nnp_nproc = 1
     with open("nnp-training.out","w") as fpout:
         subprocess.run([training_exe],stdout=fpout,stderr=subprocess.STDOUT)

def run_predict():
     '''Run nnp-predict
     '''
     n2p2_root = os.environ.get("N2P2_ROOT")
     if not n2p2_root:
         predict_exe = "nnp-predict"
     else:
         predict_exe = Path(n2p2_root) / "bin" / "nnp-predict"
     predict_exe = str(predict_exe)
     nnp_nproc = 1
     with open("nnp-predict.out","w") as fpout:
         subprocess.run([predict_exe,"0"],stdout=fpout,stderr=subprocess.STDOUT)

def write_input(elements: list[str], cutoff_type: int, cutoff_alpha: float, counts: list[int]) -> None:
     '''Write an input file

     This function writes an input file for the N2P2 tools. 

     Some of the parameters are case specific so the corresponding
     values need to be passed by the function arguments. This is 
     particularly true for the chemical elements in the system of
     interest. 

     Other characteristics we may want to set are the cutoff type
     and the cutoff radius. More on this below.

     Finally, N2P2 uses random number generators but the seed is
     specified in the input file. Here we want to use the NNP in a mode
     that is similar to DeePMD. I.e. we want to train models with the
     same hyperparameters but different initial weights to get a sense
     of the parameter uncertainty after training. That means that for
     every model we train we need a unique seed. Python's random number
     generator can be initialized with a hardware entropy pool. We'll
     use this approach to pick random random number generator seeds.

     N2P2 supports different cutoff types which are enumerated as:
     - CT_HARD  (0): No cutoff(?)
     - CT_COS   (1): (cos(pi*x)+1)/2
     - CT_TANHU (2): tanh^3(1 - r/r_c)
     - CT_TANH  (3): tanh^3(1 - r/r_c), except if r=0 then 1
     - CT_EXP   (4): exp(1 - 1/(1-x*x))
     - CT_POLY1 (5): (2x - 3)x^2 + 1
     - CT_POLY2 (6): ((15 - 6x)x - 10) x^3 + 1
     - CT_POLY3 (7): (x(x(20x - 70) + 84) - 35)x^4 + 1
     - CT_POLY4 (8): (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1
     See: n2p2/src/libnnp/CutoffFunction.h

     In general we follow the suggestions in
     https://github.com/CompPhysVienna/n2p2/blob/master/examples/input.nn.recommended
     '''
     num_elm = len(elements)
     if num_elm < 1:
         raise RuntimeError(f"N2P2 write_input: Invalid number of chemical elements: {num_elm}")
     retrain = False
     file_list  = glob.glob("weights.*.data")
     if len(file_list) > 0:
         retrain = True
     with open("input.nn","w") as fp:
         fp.write(f"number_of_elements {num_elm}\n")
         fp.write( "elements")
         for element in elements:
             fp.write(f" {element}")
         fp.write( "\n")
         fp.write(f"cutoff_type {str(cutoff_type)} {str(cutoff_alpha)}\n")
         fp.write( "scale_symmetry_functions_sigma\n")
         fp.write( "scale_min_short 0.0\n")
         fp.write( "scale_max_short 1.0\n")
         fp.write( "global_hidden_layers_short 2\n")
         fp.write( "global_nodes_short 15 15\n")
         fp.write( "global_activation_short p p l\n")
         fp.write( "use_short_forces\n")
         # The random_seed is mentioned here so we don't forget it.
         # All the parameters printed out here are general for this case.
         # The random_seed need to be set separately for every training input
         # and needs to be unique among all training runs.
         # So after generating the generic input files we append a different
         # random_seed for every training input file. Which is probably the
         # easiest way of handling this situation.
         fp.write( "#random_seed - we'll append that at the end\n")
         if retrain:
             fp.write("use_old_weights_short\n")
         fp.write( "epochs 20\n")
         fp.write( "normalize_data_set force\n")
         fp.write( "updater_type 1\n")
         fp.write( "parallel_mode 0\n")
         fp.write( "jacobian_mode 1\n")
         fp.write( "update_strategy 0\n")
         fp.write( "selection_mode 2\n")
         fp.write( "task_batch_size_energy 1\n")
         fp.write( "task_batch_size_force 1\n")
         fp.write( "memorize_symfunc_results\n")
         fp.write( "test_fraction 0.1\n")
         fp.write( "force_weight 1.0\n")
         fp.write( "short_energy_fraction 1.000\n")
         fp.write( "force_energy_ratio 3.0\n")
         fp.write( "short_energy_error_threshold 0.00\n")
         fp.write( "short_force_error_threshold 1.00\n")
         fp.write( "rmse_threshold_trials 3\n")
         fp.write( "weights_min -1.0\n")
         fp.write( "weights_max  1.0\n")
         fp.write( "main_error_metric RMSEpa\n")
         fp.write( "write_trainpoints 10\n")
         fp.write( "write_trainforces   10\n")
         fp.write( "write_weights_epoch 10\n")
         fp.write( "write_neuronstats   10\n")
         fp.write( "write_trainlog\n")
         fp.write( "kalman_type    0\n")
         fp.write( "kalman_epsilon 1.0E-2\n")
         fp.write( "kalman_q0      0.01\n")
         fp.write( "kalman_qtau    2.302\n")
         fp.write( "kalman_qmin    1.0E-6\n")
         fp.write( "kalman_eta     0.01\n")
         fp.write( "kalman_etatau  2.302\n")
         fp.write( "kalman_etamax  1.0\n")
     gen_symfunc(elements, Path("input.nn"), r_cutoff = 6.0, counts = counts)    

def append_random_seed(num: int) -> None:
    '''Append a random random seed to input.nn

    Everytime we call this function we create a new random number generator.
    This generator will be seeded from the hardware entropy pool (if available
    in your machine). Just incase there is no entropy pool we loop a number
    of times over the generator and draw a random number that we'll use as
    a seed. By ensuring we set num to a different value on every call we can
    still draw a unique seed.
    '''
    if num < 1:
        raise RuntimeError(f"append_random_seed: num must be positive: {str(num)}")
    random = np.random.default_rng()
    ival = random.integers(low=sys.maxsize)
    for ii in range(num):
       ival = random.integers(low=sys.maxsize)
    with open("input.nn","a") as fp:
        fp.write(f"random_seed {str(ival)}\n")

def create_directories(data_path: Path = None) -> None:
    '''Generate the directories for scaling and training runs
    '''
    #
    # Make directories if needed
    #
    #os.makedirs("scaling",exist_ok=True)
    os.makedirs("train-1",exist_ok=True)
    os.makedirs("train-2",exist_ok=True)
    os.makedirs("train-3",exist_ok=True)
    os.makedirs("train-4",exist_ok=True)
    #
    # Softlink the training data
    #
    if data_path is None:
        data_path = Path("..") / "ab-initio" / "input.data"
    #path = Path("scaling") / "input.data"
    #if not path.exists():
    #    subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-1") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-2") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-3") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    path = Path("train-4") / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    #
    # Create input files
    #
    elements, counts = read_elements(data_path)
    for ii in range(len(elements)):
        element = elements[ii]
        elements[ii] = element.capitalize()
    #path = Path("scaling") / "input.nn"
    #if not path.exists():
    #    os.chdir("scaling")
    #    write_input(elements,6,0.0,counts)
    #    append_random_seed(1)
    #    os.chdir("..")
    path = Path("train-1") / "input.nn"
    if not path.exists():
        os.chdir("train-1")
        write_input(elements,6,0.0,counts)
        append_random_seed(2)
        os.chdir("..")
    path = Path("train-2") / "input.nn"
    if not path.exists():
        os.chdir("train-2")
        write_input(elements,6,0.0,counts)
        append_random_seed(3)
        os.chdir("..")
    path = Path("train-3") / "input.nn"
    if not path.exists():
        os.chdir("train-3")
        write_input(elements,6,0.0,counts)
        append_random_seed(4)
        os.chdir("..")
    path = Path("train-4") / "input.nn"
    if not path.exists():
        os.chdir("train-4")
        write_input(elements,6,0.0,counts)
        append_random_seed(5)
        os.chdir("..")
    #
    # Create softlinks to scaling.data (this file will be generated when nnp-scaling is run)
    #
    #path = Path("train-1") / "scaling.data"
    #if not path.exists():
    #    os.chdir("train-1")
    #    subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
    #    os.chdir("..")
    #path = Path("train-2") / "scaling.data"
    #if not path.exists():
    #    os.chdir("train-2")
    #    subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
    #    os.chdir("..")
    #path = Path("train-3") / "scaling.data"
    #if not path.exists():
    #    os.chdir("train-3")
    #    subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
    #    os.chdir("..")
    #path = Path("train-4") / "scaling.data"
    #if not path.exists():
    #    os.chdir("train-4")
    #    subprocess.run(["ln","-s","../scaling/scaling.data","scaling.data"])
    #    os.chdir("..")

def create_directory(dir_path: Path, data_path: Path = None) -> None:
    '''Generate the directories for scaling and training runs
    '''
    #
    # Make directories if needed
    #
    os.makedirs(str(dir_path),exist_ok=True)
    #
    # Softlink the training data
    #
    if data_path is None:
        data_path = Path("..") / "ab-initio" / "input.data"
    path = Path(dir_path) / "input.data"
    if not path.exists():
        subprocess.run(["ln","-s",str(data_path),str(path)])
    #
    # Create input files
    #
    num = int(str(dir_path)[-1])
    elements, counts = read_elements(data_path)
    for ii in range(len(elements)):
        element = elements[ii]
        elements[ii] = element.capitalize()
    path = Path(dir_path) / "input.nn"
    if not path.exists():
        os.chdir(dir_path)
        write_input(elements,6,0.0,counts)
        append_random_seed(num)
        os.chdir("..")

def select_best_model():
    '''Select the "best" model for inference

    We bluntly assume that the best model is the one that has been trained
    the most, i.e. the model with the heighest epoch count.

    In the training directories there will be files with names like

      weights.001.000010.out

    these filenames map onto a pattern

      weights.N.M.out

    where N is the atomic number of chemical elements, and M is the epoch
    that corresponds to these weights. The model name for the inference
    part corresponds to weights.N.data. 

    So, here we want to establish the highest value of M and copy the files
    for all N from weights.N.M.out to weights.N.data in the current 
    directory.
    '''
    file_list    = glob.glob("weights.*.*.out")
    epoch_list   = []
    element_list = []
    for filename in file_list:
        fileinfo = filename.split(".")
        element_list.append(fileinfo[1])
        epoch_list.append(fileinfo[2])
    element_list = _sort_uniq(element_list)
    epoch_list   = _sort_uniq(epoch_list)
    last_epoch   = epoch_list[-1]
    for element in element_list:
        subprocess.run(["cp",f"weights.{element}.{last_epoch}.out",f"weights.{element}.data"])

def _sort_uniq(sequence):
    """Return a sorted sequence of unique instances.

    See https://stackoverflow.com/questions/2931672/what-is-the-cleanest-way-to-do-a-sort-plus-uniq-on-a-python-list
    """
    return list(map(operator.itemgetter(0),itertools.groupby(sorted(sequence))))
