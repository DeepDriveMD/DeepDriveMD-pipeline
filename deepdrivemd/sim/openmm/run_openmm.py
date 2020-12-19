import shutil
from pathlib import Path
from typing import Optional
import simtk.unit as u
import simtk.openmm.app as app
from deepdrivemd.config import get_config
from mdtools.openmm.sim import configure_simulation
from mdtools.openmm.reporter import OfflineReporter
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.sim.openmm.config import OpenMMConfig


class SimulationContext:
    def __init__(self, cfg: OpenMMConfig):

        self.cfg = cfg
        self.api = DeepDriveMD_API(cfg.experiment_directory)

        # Use node local storage if available. Otherwise, write to result directory.
        if cfg.node_local_path is not None:
            self.workdir = cfg.node_local_path.joinpath(cfg.dir_prefix)
        else:
            self.workdir = cfg.result_dir.joinpath(cfg.dir_prefix)

        self._init_workdir()

    @property
    def _sim_prefix(self) -> Path:
        return self.workdir.joinpath(self.cfg.dir_prefix)

    @property
    def pdb_file(self) -> str:
        return self._pdb_file.as_posix()

    @property
    def traj_file(self) -> str:
        return self._sim_prefix.with_suffix(".dcd").as_posix()

    @property
    def h5_prefix(self) -> str:
        return self._sim_prefix.as_posix()

    @property
    def log_file(self) -> str:
        return self._sim_prefix.with_suffix(".log").as_posix()

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

    def _init_workdir(self):
        """Setup workdir and copy PDB/TOP files."""

        self.workdir.mkdir()

        self._pdb_file = self._get_pdb_file()

        if self.cfg.solvent_type == "explicit":
            self._top_file = self._copy_top_file()
        else:
            self._top_file = None

    def _get_pdb_file(self) -> Path:
        if self.cfg.pdb_file is not None:
            # Initial iteration
            return self._copy_pdb_file()

        assert self.cfg.restart_point is not None
        # Iterations after outlier detection
        outlier = self.api.get_restart_pdb(self.cfg.restart_point)
        pdb_file = self.workdir.joinpath("input.pdb")
        self.api.write_pdb(
            pdb_file,
            outlier["input_pdb_file"],
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
        top_file = self.api.get_topology(self.cfg.initial_pdb_dir, Path(self.pdb_file))
        assert top_file is not None
        local_top_file = shutil.copy(top_file, self.workdir.joinpath(top_file.name))
        return Path(local_top_file)

    def move_results(self):
        shutil.move(self.workdir, self.cfg.result_dir)


def configure_reporters(
    sim: simtk.openmm.app.Simulation,  # noqa F821
    ctx: SimulationContext,
    cfg: OpenMMConfig,
    report_steps: int,
    frames_per_h5: int,
):
    # Configure DCD file reporter
    sim.reporters.append(app.DCDReporter(ctx.traj_file, report_steps))

    # Configure contact map reporter
    sim.reporters.append(
        OfflineReporter(
            ctx.h5_prefix,
            report_steps,
            frames_per_h5=frames_per_h5,
            wrap_pdb_file=ctx.pdb_file if cfg.wrap else None,
            reference_pdb_file=ctx.reference_pdb_file,
            openmm_selection=cfg.openmm_selection,
            mda_selection=cfg.mda_selection,
            threshold=cfg.threshold,
            contact_map=cfg.contact_map,
            point_cloud=cfg.point_cloud,
            fraction_of_contacts=cfg.fraction_of_contacts,
        )
    )

    # Configure simulation output log
    sim.reporters.append(
        app.StateDataReporter(
            ctx.log_file,
            report_steps,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )


def run_simulation(cfg: OpenMMConfig):

    # openmm typed variables
    dt_ps = cfg.dt_ps * u.picoseconds
    report_interval_ps = cfg.report_interval_ps * u.picoseconds
    simulation_length_ns = cfg.simulation_length_ns * u.nanoseconds
    temperature_kelvin = cfg.temperature_kelvin * u.kelvin

    # Handle files
    ctx = SimulationContext(cfg)

    # Create openmm simulation object
    sim = configure_simulation(
        pdb_file=ctx.pdb_file,
        top_file=ctx.top_file,
        solvent_type=cfg.solvent_type,
        gpu_index=0,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
        heat_bath_friction_coef=cfg.heat_bath_friction_coef,
    )

    # Write all frames to a single HDF5 file
    frames_per_h5 = int(simulation_length_ns / report_interval_ps)
    # Steps between reporting DCD frames and logs
    report_steps = int(report_interval_ps / dt_ps)
    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    # Configure reporters to write output files
    configure_reporters(sim, ctx, cfg, report_steps, frames_per_h5)

    # Run simulation for nsteps
    sim.step(nsteps)

    # Move simulation data to persistent storage
    if cfg.node_local_path is not None:
        ctx.move_results()


if __name__ == "__main__":
    cfg = get_config()
    cfg = OpenMMConfig.from_yaml(**cfg)
    run_simulation(cfg)
