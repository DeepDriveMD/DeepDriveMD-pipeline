import itertools
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import parmed as pmd  # type: ignore
import simtk.openmm as omm  # type: ignore
import simtk.unit as u  # type: ignore
from lockfile import LockFile  # type: ignore
from mdtools.openmm.sim import configure_simulation  # type: ignore

from deepdrivemd.data.api import DeepDriveMD_API

# from deepdrivemd.sim.openmm.run_openmm import SimulationContext
from deepdrivemd.sim.openmm_stream.config import OpenMMConfig
from deepdrivemd.sim.openmm_stream.openmm_reporter import ContactMapReporter
from deepdrivemd.utils import Timer, parse_args


class SimulationContext:
    def __init__(self, cfg: OpenMMConfig):

        self.cfg = cfg
        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self._prefix = self.api.molecular_dynamics_stage.unique_name(cfg.output_path)
        self._top_file: Optional[Path] = None

        # Use node local storage if available. Otherwise, write to output directory.
        if cfg.node_local_path is not None:
            self.workdir = cfg.node_local_path.joinpath(self._prefix)
        else:
            self.workdir = cfg.output_path

        self._init_workdir()

    @property
    def _sim_prefix(self) -> Path:
        return self.workdir.joinpath(self._prefix)

    @property
    def pdb_file(self) -> str:
        return self._pdb_file.as_posix()

    @property
    def top_file(self) -> Optional[str]:
        if self._top_file is None:
            return None
        return self._top_file.as_posix()

    @property
    def reference_pdb_file(self) -> Optional[str]:
        if self.cfg.reference_pdb_file is None:
            return None
        return self.cfg.reference_pdb_file.as_posix()

    def _init_workdir(self) -> None:
        """Setup workdir and copy PDB/TOP files."""

        self.workdir.mkdir(exist_ok=True)

        self._pdb_file = self._get_pdb_file()

        if self.cfg.solvent_type == "explicit":
            self._top_file = self._copy_top_file()
        else:
            self._top_file = None

    def _get_pdb_file(self) -> Path:
        if self.cfg.pdb_file is not None:
            # Initial iteration
            return self._copy_pdb_file()

        # Iterations after outlier detection
        outlier = self.api.get_restart_pdb(self.cfg.task_idx, self.cfg.stage_idx - 1)
        system_name = self.api.get_system_name(outlier["structure_file"])
        pdb_file = self.workdir.joinpath(f"{system_name}__{self._prefix}.pdb")
        self.api.write_pdb(
            pdb_file,
            outlier["structure_file"],
            outlier["traj_file"],
            outlier["frame"],
            self.cfg.in_memory,
        )
        return pdb_file

    def _copy_pdb_file(self) -> Path:
        assert self.cfg.pdb_file is not None
        copy_to_file = self.api.get_system_pdb_name(self.cfg.pdb_file)
        local_pdb_file = shutil.copy(
            self.cfg.pdb_file, self.workdir.joinpath(copy_to_file)
        )
        return Path(local_pdb_file)

    def _copy_top_file(self) -> Path:
        assert self.cfg.top_suffix is not None
        top_file = self.api.get_topology(
            self.cfg.initial_pdb_dir, Path(self.pdb_file), self.cfg.top_suffix
        )
        assert top_file is not None
        local_top_file = shutil.copy(top_file, self.workdir.joinpath(top_file.name))
        return Path(local_top_file)

    def move_results(self) -> None:
        if self.workdir != self.cfg.output_path:
            for p in self.workdir.iterdir():
                shutil.move(str(p), str(self.cfg.output_path.joinpath(p.name)))


def configure_reporters(
    sim: omm.app.Simulation,
    ctx: SimulationContext,
    cfg: OpenMMConfig,
    report_steps: int,
) -> None:

    sim.reporters.append(ContactMapReporter(report_steps, cfg))


def next_outlier(
    cfg: OpenMMConfig, sim: omm.app.Simulation
) -> Union[Tuple[Path, Path, float, str], None]:
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

    with open(cfg.current_dir / "rmsd.txt", "w") as f:  # type: ignore
        f.write(f"{rmsd}\n")  # type: ignore

    positions_pdb = cfg.current_dir / f"{md5}.pdb"
    velocities_npy = cfg.current_dir / f"{md5}.npy"

    return positions_pdb, velocities_npy, rmsd, md5


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

    outlier = next_outlier(cfg, sim)
    if outlier is not None:
        print("There are outliers")
        positions_pdb, velocities_npy, rmsd, md5 = outlier
        while True:
            try:
                positions = pmd.load_file(str(positions_pdb)).positions
                velocities = np.load(str(velocities_npy))  # type: ignore
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


def init_input(cfg: OpenMMConfig) -> None:
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


def run_simulation(cfg: OpenMMConfig) -> None:
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


def adios_configuration(cfg: OpenMMConfig) -> None:
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
