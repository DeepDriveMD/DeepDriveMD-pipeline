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
from typing import Dict, Union
import pandas as pd
import adios2


def configure_reporters(
    sim: omm.app.Simulation,
    # ctx: SimulationContext,
    cfg: OpenMMConfig,
    report_steps: int,
    iteration: int = 0,
):
    cfg.reporter = ContactMapReporter(report_steps, cfg)
    sim.reporters.append(cfg.reporter)
    """
    log_file = os.path.dirname(ctx.log_file) + f"/{iteration}/" + os.path.basename(ctx.log_file)
    sim.reporters.append(
        omm.app.StateDataReporter(
            log_file,
            report_steps,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )
    """


def next_outlier(
    cfg: OpenMMConfig, sim: omm.app.Simulation
) -> Dict[str, Union[int, float, str, np.array]]:
    """Get the next outlier to use as an initial state.

    Parameters
    ----------
    cfg : OpenMMConfig
    sim : omm.app.Simulation

    Returns
    -------
    Dict[str, Union[np.array, str, int, float, None]]
        path to pdb file with positions, path to numpy file with velocities, rmsd, md5sum, etc. depending on configuration.
        The key describes what is returned.

    """

    cfg.pickle_db = cfg.outliers_dir / "OutlierDB.pickle"

    if not os.path.exists(cfg.pickle_db):
        return None

    if cfg.lock == "set_by_deepdrivemd":
        cfg.lock = LockFile(cfg.pickle_db)

    while True:
        try:
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
            if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
                task = cfg.outliers_dir / f"{md5}.txt"
                shutil.copy(task, cfg.current_dir)
                copied_task = cfg.current_dir / f"{md5}.txt"
            cfg.lock.release()
        except Exception as e:
            print("=" * 30)
            print(e)
            sleeptime = random.randint(3, 15)
            print(f"Sleeping for {sleeptime} seconds")
            print(subprocess.getstatusoutput(f"ls -l {cfg.outliers_dir}")[1])
            print(subprocess.getstatusoutput(f"md5sum {cfg.outliers_dir}/*")[1])
            print("=" * 30)
            sys.stdout.flush()
            time.sleep(sleeptime)
            continue
        break

    with open(cfg.current_dir / "rmsd.txt", "w") as f:
        f.write(f"{rmsd}\n")

    positions_pdb = cfg.current_dir / f"{md5}.pdb"
    velocities_npy = cfg.current_dir / f"{md5}.npy"

    outputs = {
        "positions_pdb": positions_pdb,
        "velocities_npy": velocities_npy,
        "rmsd": rmsd,
        "md5": md5,
    }

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        with open(copied_task) as f:
            task_id = int(f.read())
        cfg.ligand = task_id
        outputs["ligand"] = task_id

    return outputs


def prepare_simulation(
    cfg: OpenMMConfig, iteration: int, sim: omm.app.Simulation
) -> bool:
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
    print("In prepare_simulation cfg.current_dir = ", str(cfg.current_dir))

    if sim is None:
        outlier = None
    else:
        outlier = next_outlier(cfg, sim)

    if outlier is not None:
        print("There are outliers")

        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            positions_pdb, velocities_npy, ligand = (
                outlier["positions_pdb"],
                outlier["velocities_npy"],
                outlier["ligand"],
            )
            print("ligand=", ligand)
        else:
            positions_pdb, velocities_npy = (
                outlier["positions_pdb"],
                outlier["velocities_npy"],
            )
        """
        cfg1 = cfg.copy()
        in_dir = sim_dir / "input" / "system"
        in_dir.mkdir(parents=True, exist_ok=True)
        subprocess.getstatusoutput("ln -s {cfg.top_file1} {str(in_dir)/comp.top")
        outlier_file = sim_dir / outlier["positions_pdb"]
        subprocess.getstatusoutput("ln -s {str(outlier_file} {str(in_dir}/comp.pdb}")
        cfg1.initial_pdb_dir = str(sim_dir / "input")
        cfg1.top_file1 = in_dir / "comp.top"
        """

        """
        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            init_multi_ligand(cfg, ligand)
        else:
            init_input(cfg)
        """

        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            with Timer("molecular_dynamics_SimulationContext"):
                print("cfg.pdb_file = ", cfg.pdb_file)
                print("cfg.top_file1 = ", cfg.top_file1)
                ctx = SimulationContext(cfg)
                print("ctx = ", ctx)
                print("dir(ctx) = ", dir(ctx))
                print("ctx.pdb_file = ", ctx.pdb_file)
                print("ctx.top_file = ", ctx.top_file)

        while True:
            try:
                positions = pmd.load_file(str(positions_pdb)).positions
                # positions = pmd.load_file(ctx.top_file, xyz=str(positions_pdb)).positions
                velocities = np.load(str(velocities_npy))
                break
            except Exception as e:
                print("Exception ", e)
                print(f"Waiting for {positions_pdb} and {velocities_npy}")
                time.sleep(5)

        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            with Timer("molecular_dynamics_configure_simulation"):
                dt_ps = cfg.dt_ps * u.picoseconds
                temperature_kelvin = cfg.temperature_kelvin * u.kelvin
                print("positions_pdb = ", positions_pdb)
                print("ctx.top_file = ", ctx.top_file)
                # cfg.pdb_file = str(positions_pdb)
                try:
                    del sim
                except Exception as e:
                    print(e)

                sim = configure_simulation(
                    pdb_file=ctx.pdb_file,  # str(positions_pdb),  # ctx.pdb_file,
                    top_file=ctx.top_file,
                    solvent_type=cfg.solvent_type,
                    gpu_index=0,
                    dt_ps=dt_ps,
                    temperature_kelvin=temperature_kelvin,
                    heat_bath_friction_coef=cfg.heat_bath_friction_coef,
                )
        with Timer("molecular_dynamics_configure_reporters"):
            try:
                sim.reporters.pop()
            except Exception as e:
                print(e)
            configure_reporters(sim, cfg, cfg.report_steps, iteration)

        """
        print("="*20)
        print(help(sim.context.setPositions))
        print("="*20)
        print(dir(sim.context))
        print("="*20)
        """

        sim.context.setPositions(positions)

        if random.random() < cfg.copy_velocities_p:
            print("Copying velocities from outliers")
            sim.context.setVelocities(velocities)
        else:
            print("Generating velocities randomly")
            sim.context.setVelocitiesToTemperature(
                cfg.temperature_kelvin * u.kelvin, random.randint(1, 10000)
            )

        """
        sim.context.setVelocitiesToTemperature(
            cfg.temperature_kelvin * u.kelvin, random.randint(1, 10000)
        )
        """

        return True, sim
    else:
        print("There are no outliers")

        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            init_multi_ligand(cfg)
        else:
            init_input(cfg)

        with Timer("molecular_dynamics_SimulationContext"):
            ctx = SimulationContext(cfg)
            print("ctx = ", ctx)
            print("dir(ctx) = ", dir(ctx))

        with Timer("molecular_dynamics_configure_simulation"):
            dt_ps = cfg.dt_ps * u.picoseconds
            temperature_kelvin = cfg.temperature_kelvin * u.kelvin
            try:
                del sim
            except Exception as e:
                print(e)
                pass

            sim = configure_simulation(
                pdb_file=ctx.pdb_file,
                top_file=ctx.top_file,
                solvent_type=cfg.solvent_type,
                gpu_index=0,
                dt_ps=dt_ps,
                temperature_kelvin=temperature_kelvin,
                heat_bath_friction_coef=cfg.heat_bath_friction_coef,
                # explicit_barostat="MonteCarloAnisotropicBarostat",  ### ifs
            )
            sim.context.setVelocitiesToTemperature(
                cfg.temperature_kelvin * u.kelvin, random.randint(1, 10000)
            )

        with Timer("molecular_dynamics_configure_reporters"):
            configure_reporters(sim, cfg, cfg.report_steps, iteration)

        return False, sim


def prepare_simulation1(
    cfg: OpenMMConfig, iteration: int, sim: omm.app.Simulation
) -> bool:
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
    os.chdir(str(sim_dir))
    print("In prepare_simulation cfg.current_dir = ", str(cfg.current_dir))

    outlier = next_outlier(cfg, sim)
    if outlier is not None:
        print("There are outliers")
        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            positions_pdb, velocities_npy, ligand = (
                outlier["positions_pdb"],
                outlier["velocities_npy"],
                outlier["ligand"],
            )
        else:
            positions_pdb, velocities_npy = (
                outlier["positions_pdb"],
                outlier["velocities_npy"],
            )
        while True:
            try:
                positions = pmd.load_file(str(positions_pdb)).positions
                velocities = np.load(str(velocities_npy))
                break
            except Exception as e:
                print("Exception ", e)
                print(f"Waiting for {positions_pdb} and {velocities_npy}")
                time.sleep(5)

        """Is it needed?
           Should it be on top?
           Should there will be also
           else:
              init(cfg)
        """
        if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
            init_multi_ligand(cfg, ligand)

        """Should there be
           del ctx
           before creating a new one?
        """
        with Timer("molecular_dynamics_SimulationContext"):
            ctx = SimulationContext(cfg)
            print("ctx = ", ctx)
            print("dir(ctx) = ", dir(ctx))

        with Timer("molecular_dynamics_configure_simulation"):
            dt_ps = cfg.dt_ps * u.picoseconds
            temperature_kelvin = cfg.temperature_kelvin * u.kelvin
            print("positions_pdb = ", positions_pdb)
            print("ctx.top_file = ", ctx.top_file)
            try:
                del sim
            except Exception as e:
                print(e)
                pass
            sim = configure_simulation(
                pdb_file=str(positions_pdb),  # ctx.pdb_file,
                top_file=ctx.top_file,
                solvent_type=cfg.solvent_type,
                gpu_index=0,
                dt_ps=dt_ps,
                temperature_kelvin=temperature_kelvin,
                heat_bath_friction_coef=cfg.heat_bath_friction_coef,
            )
            print("dir(sim)=", dir(sim))
            print("type(sim)=", type(sim))
            print("sim=", sim)
        with Timer("molecular_dynamics_configure_reporters"):
            configure_reporters(sim, ctx, cfg, cfg.report_steps, iteration)

        sim.context.setPositions(positions)
        if random.random() < cfg.copy_velocities_p:
            print("Copying velocities from outliers")
            sim.context.setVelocities(velocities)
        else:
            print("Generating velocities randomly")
            sim.context.setVelocitiesToTemperature(
                cfg.temperature_kelvin * u.kelvin, random.randint(1, 10000)
            )
        return True, sim
    else:
        print("There are no outliers")

        with Timer("molecular_dynamics_SimulationContext"):
            ctx = SimulationContext(cfg)
            print("ctx = ", ctx)
            print("dir(ctx) = ", dir(ctx))

        with Timer("molecular_dynamics_configure_simulation"):
            dt_ps = cfg.dt_ps * u.picoseconds
            temperature_kelvin = cfg.temperature_kelvin * u.kelvin
            try:
                del sim
            except Exception as e:
                print(e)
                pass
            sim = configure_simulation(
                pdb_file=ctx.pdb_file,
                top_file=ctx.top_file,
                solvent_type=cfg.solvent_type,
                gpu_index=0,
                dt_ps=dt_ps,
                temperature_kelvin=temperature_kelvin,
                heat_bath_friction_coef=cfg.heat_bath_friction_coef,
            )
            """Should velocities be randomly generated?"""
        with Timer("molecular_dynamics_configure_reporters"):
            configure_reporters(sim, ctx, cfg, cfg.report_steps, iteration)

        return False, sim


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


def init_multi_ligand(cfg: OpenMMConfig, task_id=None):
    if task_id is None:
        task_id = cfg.task_idx
    table = pd.read_csv(cfg.multi_ligand_table)
    pdb = table["pdb"][task_id]
    tdir = table["tdir"][task_id]
    cfg.pdb_file = f"{tdir}/system/{pdb}"
    cfg.initial_pdb_dir = tdir
    print(
        f"init_multi_ligand: id = {task_id}, pdb = {cfg.pdb_file}, tdir = {cfg.initial_pdb_dir}"
    )
    cfg.ligand = task_id  # cfg.task_idx


def run_simulation(cfg: OpenMMConfig):

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        init_multi_ligand(cfg)
    else:
        init_input(cfg)

    # openmm typed variables
    dt_ps = cfg.dt_ps * u.picoseconds
    report_interval_ps = cfg.report_interval_ps * u.picoseconds
    simulation_length_ns = cfg.simulation_length_ns * u.nanoseconds
    # temperature_kelvin = cfg.temperature_kelvin * u.kelvin

    """
    sim_dir = cfg.output_path / "0"
    sim_dir.mkdir(exist_ok=True)
    cfg.current_dir = sim_dir

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

    """

    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    report_steps = int(report_interval_ps / dt_ps)
    cfg.report_steps = report_steps
    print("report_steps = ", report_steps)
    """
    # Configure reporters to write output files
    with Timer("molecular_dynamics_configure_reporters"):
        configure_reporters(sim, ctx, cfg, report_steps)

    """

    _, sim = prepare_simulation(cfg, 0, None)

    # Infinite simulation loop
    for iteration in itertools.count(0):
        # Run simulation for nsteps
        print(f"Simulation iteration {iteration}")
        sys.stdout.flush()

        with Timer("molecular_dynamics_step"):
            sim.step(nsteps)

        subprocess.getstatusoutput(f"touch {str(cfg.current_dir)}/done")

        _, sim = prepare_simulation(cfg, iteration + 1, sim)


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
    print("args.config = ", args.config)
    sys.stdout.flush()
    cfg = OpenMMConfig.from_yaml(args.config)
    adios_configuration(cfg)
    cfg.bp_file = cfg.output_path / "md.bp"

    stream_name = os.path.basename(cfg.output_path)
    cfg._adios_stream = adios2.open(
        name=str(cfg.bp_file),
        mode="w",
        config_file=str(cfg.adios_cfg),
        io_in_config_file=stream_name,
    )

    run_simulation(cfg)

    cfg._adios_stream.close()
