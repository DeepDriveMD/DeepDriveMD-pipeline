import shutil
import simtk.unit as u
import simtk.openmm as omm
from mdtools.openmm.sim import configure_simulation
from deepdrivemd.utils import Timer, parse_args
from deepdrivemd.sim.openmm_stream.config import OpenMMConfig
from deepdrivemd.sim.openmm_stream.openmm_reporter import ContactMapReporter
import sys
import os
import time
import random
from lockfile import LockFile
import pickle
import parmed as pmd
import numpy as np
import subprocess
import itertools
from deepdrivemd.sim.openmm.run_openmm import SimulationContext


def configure_reporters(
    sim: omm.app.Simulation,
    ctx: SimulationContext,
    cfg: OpenMMConfig,
    report_steps: int,
):

    sim.reporters.append(ContactMapReporter(report_steps, cfg))


def next_outlier(cfg: OpenMMConfig, sim: omm.app.Simulation):
    """Get the next outlier to use as an initial state.

    Parameters
    ----------
    cfg : OpenMMConfig
    sim : omm.app.Simulation

    Returns
    -------
    Tuple[str, str, float, str]
        path to pdb file with positions, path to numpy file with velocities, rmsd, md5sum

    """

    cfg.pickle_db = cfg.outliers_dir / "OutlierDB.pickle"

    print(f"cfg.outliers_dir = {cfg.outliers_dir}")
    print(f"cfg.pickle_db = {cfg.pickle_db}")
    print(f"cfg.current_dir = {cfg.current_dir}")
    import sys

    sys.stdout.flush()

    if not os.path.exists(cfg.pickle_db):
        return None

    if cfg.lock == "set_by_deepdrivemd":
        cfg.lock = LockFile(cfg.pickle_db)

    cfg.lock.acquire()
    with open(cfg.pickle_db, "rb") as f:
        db = pickle.load(f)
    md5 = db.sorted_index[cfg.task_idx]
    rmsd = db.dictionary[md5]
    positions_pdb = cfg.outliers_dir / f"{md5}.pdb"
    velocities_npy = cfg.outliers_dir / f"{md5}.npy"
    shutil.copy(positions_pdb, cfg.current_dir)
    shutil.copy(velocities_npy, cfg.current_dir)
    shutil.copy(cfg.pickle_db, cfg.current_dir)
    cfg.lock.release()

    with open(cfg.current_dir / "rmsd.txt", "w") as f:
        f.write(f"{rmsd}\n")

    positions_pdb = cfg.current_dir / f"{md5}.pdb"
    velocities_npy = cfg.current_dir / f"{md5}.npy"

    return positions_pdb, velocities_npy, rmsd, md5


def prepare_simulation(cfg: OpenMMConfig, iteration: int, sim: omm.app.Simulation):
    """Replace positions and, with `cfg.copy_velocities_p` probability, velocities
    of the current simulation state from an outlier

    Parameters
    ----------
    cfg : OpenMMConfig
    iteration : int
    sim: omm.app.Simulation

    Returns
    -------
    bool
         True if there is an outlier, False - otherwise
    """
    sim_dir = cfg.output_path / str(iteration)
    sim_dir.mkdir(exist_ok=True)
    cfg.current_dir = sim_dir

    outlier = next_outlier(cfg, sim)
    if outlier is not None:
        print("There are outliers")
        positions_pdb, velocities_npy, rmsd, md5 = outlier
        while True:
            try:
                positions = pmd.load_file(positions_pdb).positions
                velocities = np.load(velocities_npy)
                break
            except Exception as e:
                print("Exception ", e)
                print(f"Waiting for {positions_pdb} and {velocities_npy}")
                time.sleep(5)

        sim.context.setPositions(positions)
        if random.random() < cfg.copy_velocities_p:
            print("Copying velocities from outliers")
            sim.context.setVelocities(velocities)
        else:
            print("Generating velocities randomly")
            sim.context.setVelocitiesToTemperature(
                cfg.temperature_kelvin * u.kelvin, random.randint(1, 10000)
            )
        return True
    else:
        print("There are no outliers")
        return False


def init_input(cfg: OpenMMConfig):
    """The first iteration of the simulation is initialized from pdb
    files in `cfg.initial_pdb_dir`. For the given simulation the pdb file is
    selected using simulation `task_id` in a round robin fashion.
    """
    pdb_files = list(cfg.initial_pdb_dir.glob("*.pdb")) + list(
        cfg.initial_pdb_dir.glob("*/*.pdb")
    )
    pdb_files.sort()
    n = len(pdb_files)
    i = int(cfg.task_idx) % n
    cfg.pdb_file = pdb_files[i]
    print(f"init_input: n = {n}, i = {i}, pdb_file = {cfg.pdb_file}")


def run_simulation(cfg: OpenMMConfig):
    init_input(cfg)

    # openmm typed variables
    dt_ps = cfg.dt_ps * u.picoseconds
    report_interval_ps = cfg.report_interval_ps * u.picoseconds
    simulation_length_ns = cfg.simulation_length_ns * u.nanoseconds
    temperature_kelvin = cfg.temperature_kelvin * u.kelvin

    # Handle files
    with Timer("molecular_dynamics_SimulationContext"):
        ctx = SimulationContext(cfg)

    # Create openmm simulation object
    with Timer("molecular_dynamics_configure_simulation"):
        sim = configure_simulation(
            pdb_file=ctx.pdb_file,
            top_file=ctx.top_file,
            solvent_type=cfg.solvent_type,
            gpu_index=0,
            dt_ps=dt_ps,
            temperature_kelvin=temperature_kelvin,
            heat_bath_friction_coef=cfg.heat_bath_friction_coef,
        )

    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    report_steps = int(report_interval_ps / dt_ps)
    print("report_steps = ", report_steps)

    # Configure reporters to write output files
    with Timer("molecular_dynamics_configure_reporters"):
        configure_reporters(sim, ctx, cfg, report_steps)

    # Infinite simulation loop
    for iteration in itertools.count(0):
        # Run simulation for nsteps
        print(f"Simulation iteration {iteration}")
        sys.stdout.flush()
        with Timer("molecular_dynamics_step"):
            sim.step(nsteps)
        prepare_simulation(cfg, iteration, sim)


def adios_configuration(cfg: OpenMMConfig):
    """Read a template `adios.xml` file, replace `SimulationOutput`
    stream name with the simulation directory and write the resulting
    configuration file into simulation directory.
    """
    cfg.adios_cfg = cfg.output_path / "adios.xml"
    shutil.copy(cfg.adios_xml_sim, cfg.adios_cfg)
    taskdir = os.path.basename(cfg.output_path)
    with open(cfg.adios_cfg, "r") as f:
        textxml = f.read()
    textxml = textxml.replace("SimulationOutput", taskdir)
    with open(cfg.adios_cfg, "w") as f:
        f.write(textxml)


if __name__ == "__main__":
    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()
    args = parse_args()
    cfg = OpenMMConfig.from_yaml(args.config)
    adios_configuration(cfg)
    cfg.bp_file = cfg.output_path / "md.bp"
    run_simulation(cfg)
