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

def make_nwchemrc(workdir: PathLike, nwchem_top: PathLike) -> None:
    '''
    Create an nwchemrc file if needed

    Check whether the nwchemrc file exists. If it does
    not, create one with the appropiate paths for the
    force field parameters. The force field parameters
    live under nwchem_top/src/data/.
    '''
    if (Path("/etc/nwchemrc").is_file() or 
        Path("~/.nwchemrc").is_file() or
        Path(workdir.joinpath("nwchemrc")).is_file()):
        # The nwchemrc already exists, so we are done
        return
    if not nwchem_top:
        nwchem_top = Path(os.environ.get("NWCHEM_TOP"))
    else:
        nwchem_top = Path(nwchem_top)
    if nwchem_top:
        nwchem_data = nwchem_top.joinpath("src/data")
    else:
        raise RuntimeError("make_nwchemrc: NWCHEM_TOP undefined")
    fp = open(workdir.joinpath("nwchemrc"),"w")
    fp.write("ffield amber\n")
    # We cannot use joinpath here as that would strip the trailing "/" off.
    # In NWChem the trailing "/" indicates a directory instead of a file,
    # i.e. "stuff" is a file whereas "stuff/" is a directory.
    fp.write("amber_1 "+str(nwchem_data)+"/amber_s/\n")
    fp.write("amber_2 "+str(nwchem_data)+"/amber_x/\n")
    fp.write("amber_3 "+str(nwchem_data)+"/amber_t/\n")
    fp.write("amber_4 "+str(nwchem_data)+"/amber_q/\n")
    fp.write("amber_5 "+str(nwchem_data)+"/amber_u/\n")
    fp.write("spce    "+str(nwchem_data)+"/solvents/spce.rst\n")
    fp.close()

def run_nwchem(nwchem_top: PathLike, tag: str) -> None:
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
    fp = open("nwchemdat"+str(tag)+".out","w")
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
    # Need to copy the PDB file because NWChem accepts filenames of only 80 characters at most.
    subprocess.run(["cp",str(pdb),"nwchemdat_input.pdb"])
    fix_input_pdb("nwchemdat_input.pdb")
    fp = open("nwchemdat.nw","w")
    fp.write("echo\n")
    fp.write("start nwchemdat\n")
    fp.write("prepare\n")
    fp.write("  system nwchemdat_md\n")
    # make a new restart file
    fp.write("  new_rst\n")
    # make a new topology file and sequence file
    fp.write("  new_top new_seq\n")
    fp.write("  source nwchemdat_input.pdb\n")
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

def gen_input_dynamics(do_md: bool, md_dt_ps: float, md_time_ns: float, temperature_K: float, report_interval_ps: float) -> None:
    '''
    Generate input for minimization

    do_md:              if True generate input for proper MD run
                        else generate input for equilibration
    md_dt_ps:           the time step in picoseconds
    md_time_ns:         the simulation time in nanoseconds
    temperature_K:      the temperature in Kelvin
    report_interval_ps: the time between writing the system coordinates
                        to the trajectory file
    '''
    if not md_dt_ps:
        raise RuntimeError("gen_input_dynamics: undefined timestep")
    if not md_time_ns:
        raise RuntimeError("gen_input_dynamics: undefined simulation time")
    if not temperature_K:
        raise RuntimeError("gen_input_dynamics: undefined temperature")
    if not report_interval_ps:
        raise RuntimeError("gen_input_dynamics: undefined report interval")
    numsteps = max(int((md_time_ns*1000)/md_dt_ps),1)
    nreport = max(int(report_interval_ps/md_dt_ps),1)
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
    # the Berendsen thermostat is susceptable to the flying ice cube syndrome
    # suppress translational and rotational motion to avoid this.
    fp.write("  update motion 100\n")
    fp.write("  print extra out6\n")
    fp.write("  record rest "+str(numsteps)+"\n")
    if do_md:
        fp.write("  record scoor "+str(nreport)+" ascii\n")
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
    fp.write("  copy solute nwchemdat_md.xyz\n")
    #fp.write("  copy nwchemdat_md.xyz\n")
    fp.write("end\n")
    fp.write("task analysis\n")
    fp.close()

def fix_nwchem_xyz(xyz_file: PathLike) -> None:
    '''
    The NWChem MD Analysis module writes broken XYZ files that need fixing

    For the solute atoms the analysis module in nwchem writes:

        Chemical_Symbol X, Y, Z

    This is invalid and causes problems with Python based XYZ readers.
    The correct format is

        Chemical_Symbol X  Y  Z

    I.e. there should be no commas.
    This function replaces all the commas with spaces to fix this
    issue.

    Depending on the values of the coordinates NWChem may write the
    coordinates in a funky Fortran way. E.g. the code may write

        H -3.95 2*8.81

    instead of 

        H -3.95  8.81  8.81

    I think even Fortran cannot read this data format.
    This function also detects this "*"-notation and converts it
    back to compliant XYZ.
    '''
    if not [ [Path(xyz_file).suffix == ".xyz"] or [Path(xyz_file).suffix == ".XYZ"] ]:
        return
    fp = open(xyz_file,"r")
    in_xyz = fp.readlines()
    fp.close()
    out_xyz = []
    for line in in_xyz:
        # Fix any comma-s
        line1 = line.replace(","," ")
        # Fix any *-notation stuff
        if "*" in line1:
            tokens = line1.split()
            # Keep the element symbol
            line2 = tokens[0]
            # Process the coordinates
            for token in tokens[1:]:
                if "*" in token:
                    tokens2 = token.split("*")
                    n = int(tokens2[0])
                    for i in range(0,n):
                        line2 = line2 + " " + tokens2[1]
                else:
                    line2 = line2 + " " + token
            line2 = line2 + "\n"
        else:
            line2 = line1
        out_xyz.append(line2)
    fp = open(xyz_file,"w")
    fp.writelines(out_xyz)
    fp.close()

def fix_input_pdb(pdb_file: PathLike) -> None:
    '''
    Fix the input PDB files that MDAnalysis produces

    The MDAnalysis package typically fails to read the CRYST1 line
    in a PDB file. The package considers that this is not a problem
    as most PDB file will list lattice vectors with lengths of 1.0
    anyway. However, for MD programs this is often a problem as they
    use the CRYST1 line to define the simulation box. A box of 
    1 Angstrom cubed is way too small and causes calculations to fail
    as the solute cannot be solvated in such a tiny box. 

    This function addresses this problem by reading the PDB file,
    finding the minimum and maximum x-, y-, and z-coordinates,
    calculate the minimum box edges required, add 7.5 Angstrom
    of padding and generates a cubic box of the largest edge length.
    The CRYST1 line is then updated with these dimensions and the PDB
    file saved.
    '''
    if not [ [Path(pdb_file).suffix == ".pdb"] or [Path(pdb_file).suffix == ".PDB"] ]:
        return
    fp = open(pdb_file,"r")
    in_pdb = fp.readlines()
    fp.close()
    out_pdb = []
    x_min =  1.0e12
    x_max = -1.0e12
    y_min =  1.0e12
    y_max = -1.0e12
    z_min =  1.0e12
    z_max = -1.0e12
    for line in in_pdb:
        if line[:6] == "HETATM" or line[:6] == "ATOM  ":
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if x < x_min:
                x_min = x
            if x_max < x:
                x_max = x
            if y < y_min:
                y_min = y
            if y_max < y:
                y_max = y
            if z < z_min:
                z_min = z
            if z_max < z:
                z_max = z
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    length = x_len
    if y_len > length:
        length = y_len
    if z_len > length:
        length = z_len
    length += 7.5 # This length cubed is the final box size
    for line in in_pdb:
        if line[:6] == "CRYST1":
            out_pdb.append(f"CRYST1{length:9.3f}{length:9.3f}{length:9.3f}  90.00  90.00  90.00 P 1           1\n")
        else:
            out_pdb.append(line)
    fp = open(pdb_file,"w")
    fp.writelines(out_pdb)
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
