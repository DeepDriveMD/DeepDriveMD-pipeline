'''
Define the set up to run an NWChem MD simulation

MD simulations in NWChem go through a number of stages:
    - Generate an nwchemrc file detailing where the
      parameter files reside
    - Creating a topology file
    - Relaxing the initial structure
    - Running MD simulations
    - Analyzing the results
Here we provide functions for each of these stages. For
actual runs an number of these phases might be combined
so that ultimately we have only three phases:
    - Initialization
    - Simulation
    - Analysis
'''
import os
from os import PathLike
import subprocess
import MDAnalysis
from pathlib import Path

def make_nwchemrc(nwchem_top: PathLike) -> None:
    '''
    Create an nwchemrc file if needed

    Check whether the nwchemrc file exists. If it does
    not, create one with the appropiate paths for the
    force field parameters. The force field parameters
    live under nwchem_top/src/data/.
    '''
    if (Path("/etc/nwchemrc").is_file() or 
        Path("~/.nwchemrc").is_file() or
        Path("./nwchemrc").is_file()):
        # The nwchemrc already exists, so we are done
        return
    if not nwchem_top:
        nwchem_top = os.environ.get("NWCHEM_TOP")
    if nwchem_top:
        nwchem_data = nwchem_top+"/src/data"
    else:
        raise RuntimeError("make_nwchemrc: NWCHEM_TOP undefined")
    fp = open("./nwchemrc","w")
    fp.write("ffield amber\n")
    fp.write("amber_1 "+nwchem_data+"/amber_s/\n")
    fp.write("amber_2 "+nwchem_data+"/amber_x/\n")
    fp.write("amber_3 "+nwchem_data+"/amber_t/\n")
    fp.write("amber_4 "+nwchem_data+"/amber_q/\n")
    fp.write("amber_5 "+nwchem_data+"/amber_u/\n")
    fp.write("spce    "+nwchem_data+"/solvents/spce.rst\n")
    fp.close()

def run_nwchem(nwchem_top: PathLike) -> None:
    '''
    Run the NWChem executable in a system call

    NWChem is invoked with fixed command line parameters.
    - nwchem.nw - input file
    - nwchem.out - output file
    The executable name is constructed from NWCHEM_TOP as
    NWCHEM_TOP/bin/LINUX64/nwchem. In principle LINUX64 
    could be different for different operating systems,
    but at present LINUX64 is correct for almost any
    computer system (this used to be very different).
    '''
    if not nwchem_top:
        nwchem_top = os.environ.get("NWCHEM_TOP")
    if nwchem_top:
        nwchem_exe = nwchem_top+"/bin/LINUX64/nwchem"
    else:
        raise RuntimeError("run_nwchem: NWCHEM_TOP undefined")
    if not Path(nwchem_exe).is_file():
        raise RuntimeError("run_nwchem: NWCHEM_EXE("+nwchem_exe+") is not a file")
    fp = open("nwchemdat.out","w")
    subprocess.run([nwchem_exe,"nwchemdat.nw"],stdout=fp,stderr=subprocess.STDOUT)
    fp.close()

def replace_restart_file() -> None:
    '''
    Replace the restart file after minimization

    After the prepare stage the energy of the structure needs to be 
    minimized. The minimization produces a new restart file but with
    a different name. We need to replace the old restart file with
    the new one otherwise the dynamics run will fail.
    '''
    if Path("nwchemdat_md.qrs").is_file():
        subprocess.run(["cp","nwchemdat_md.qrs","nwchemdat_md.rst"])

def gen_input_prepare(pdb: PathLike) -> None:
    '''
    Generate the input for the PREPARE step that creates the topology and restart files

    This step is typically run in serial so it should be separate from
    steps that might run in parallel.
    '''
    if not pdb:
        raise RuntimeError("gen_input_prepare: no PDB file for structure")
    if not Path(pdb).is_file():
        raise RuntimeError("gen_input_prepare: PDB("+pdb+") is not a file")
    fp = open("nwchemdat.nw","w")
    fp.write("echo\n")
    fp.write("start nwchemdat\n")
    fp.write("prepare\n")
    fp.write("  system nwchemdat_md\n")
    # make a new restart file
    fp.write("  new_rst\n")
    # make a new topology file and sequence file
    fp.write("  new_top new_seq\n")
    fp.write("  source "+pdb+"\n")
    fp.write("  solvent name HOH model spce\n")
    fp.write("  solvate\n")
    fp.write("end\n")
    fp.write("task prepare\n")
    fp.close()

def gen_input_minimize() -> None:
    '''
    Generate input for minimization
    '''
    fp = open("nwchemdat.nw","w")
    fp.write("echo\n")
    fp.write("start nwchemdat\n")
    fp.write("md\n")
    # we need sufficient memory for the structure
    fp.write("  msa 100000\n")
    fp.write("  system nwchemdat_md\n")
    fp.write("  sd 500\n")
    fp.write("  cg 500\n")
    fp.write("  print extra out6\n")
    fp.write("end\n")
    fp.write("task md optimize\n")
    fp.close()

def gen_input_dynamics(do_md: bool, md_dt_ps: float, md_time_ps: float, temperature_K: float) -> None:
    '''
    Generate input for minimization

    do_md:         if True generate input for proper MD run
                   else generate input for equilibration
    md_dt_ps:      the time step in picoseconds
    md_time_ps:    the simulation time (assumed in picoseconds)
    temperature_K: the temperature in Kelvin
    '''
    if not md_dt_ps:
        raise RuntimeError("gen_input_dynamics: undefined timestep")
    if not md_time_ps:
        raise RuntimeError("gen_input_dynamics: undefined simulation time")
    if not temperature_K:
        raise RuntimeError("gen_input_dynamics: undefined temperature")
    numsteps = int(md_time_ps/md_dt_ps)+1
    fp = open("nwchemdat.nw","w")
    fp.write("echo\n")
    fp.write("start nwchemdat\n")
    fp.write("md\n")
    # we need sufficient memory for the structure
    fp.write("  msa 100000\n")
    fp.write("  system nwchemdat_md\n")
    if do_md:
        fp.write("  vreass "+str(numsteps)+" "+str(temperature_K)+" 1.0\n")
    else:
        fp.write("  vreass 1 "+str(temperature_K)+" 0.5\n")
    fp.write("  step "+str(md_dt_ps)+" equil 0 data "+str(numsteps)+"\n")
    fp.write("  isotherm "+str(temperature_K)+" trelax 0.1\n")
    fp.write("  print extra out6\n")
    fp.write("  record rest "+str(numsteps)+" scoor 1 svelo 1 ascii\n")
    fp.write("end\n")
    fp.write("task md dynamics\n")
    fp.close()

def gen_input_analysis() -> None:
    '''
    Convert the NWChem trajectory to common file formats

    The NWChem topology and trajectory files (.top and .trj) are stored
    in its own data format. For other tools to access this data it needs
    to be converted.

    The analysis needs to produce 2 files:
    - A file that defines the atoms (here we'll generate a PDB file)
    - A file with a time series of atom positions (here we'll generate an
      
    We assume that only the coordinates of the solute are relevant. The 
    solvent has no structure and pretty much does whatever it wants. So
    exploring the phase space of the solvent is a waste.

    NWChem seems to have a major bug in that when converting the 
    trajectory into XYZ format it also prints out the water molecules
    even if you ask for just the solute atoms. All water atoms are just
    placed at position 0,0,0.
    '''
    fp = open("nwchemdat.nw","w")
    fp.write("echo\n")
    fp.write("start nwchemdat\n")
    fp.write("analysis\n")
    fp.write("  system nwchemdat_md\n")
    fp.write("  reference nwchemdat_md.rst\n")
    fp.write("  file      nwchemdat_md.trj\n")
    #fp.write("  write  1 solute nwchemdat_md.pdb\n")
    fp.write("  write  1 nwchemdat_md.pdb\n")
    # Use a large number of frames here it will save only what there is
    fp.write("  frames 1 1000000 1\n")
    #fp.write("  copy solute nwchemdat_md.xyz\n")
    fp.write("  copy nwchemdat_md.xyz\n")
    fp.write("end\n")
    fp.write("task analysis\n")
    fp.close()

def fix_nwchem_xyz(xyz_file: PathLike) -> None:
    '''
    The NWChem MD Analysis module write broken XYZ files that need fixing

    For the solute atoms the analysis module in nwchem writes:

        Chemical_Symbol X, Y, Z

    This is invalid and causes problems with Python based XYZ readers.
    The correct format is

        Chemical_Symbol X  Y  Z

    I.e. there should be no commas.
    This function replaces all the commas with spaces to fix this
    issue.
    '''
    if not [ [Path(xyz_file).suffix == ".xyz"] or [Path(xyz_file).suffix == ".XYZ"] ]:
        return
    fp = open(xyz_file,"r")
    in_xyz = fp.readlines()
    fp.close()
    out_xyz = []
    for line in in_xyz:
        out_xyz.append(line.replace(","," "))
    fp = open(xyz_file,"w")
    fp.writelines(out_xyz)
    fp.close()

def read_trajectory(topology: PathLike, trajectory: PathLike) -> None:
    '''
    Read the NWChem trajectory file and return the structure within
    '''
    if not Path(topology).is_file():
        raise RuntimeError("read_trajectory: no topology file")
    if not Path(trajectory).is_file():
        raise RuntimeError("read_trajectory: no trajectory file")
    if Path(topology).suffix == ".xyz":
        fix_nwchem_xyz(topology)
    if Path(trajectory).suffix == ".xyz":
        fix_nwchem_xyz(trajectory)
    out = MDAnalysis.Universe(topology,trajectory)
    return out
