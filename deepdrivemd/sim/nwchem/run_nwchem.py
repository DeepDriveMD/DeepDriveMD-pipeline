import shutil
import os
import sys
import time
from pathlib import Path
from typing import Optional

import openmm
import openmm.unit as u # type: ignore[import]
import openmm.app as app  # type: ignore[import]
from mdtools.nwchem.reporter import OfflineReporter  # type: ignore[import]

from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.sim.nwchem.config import NWChemConfig
from deepdrivemd.sim.nwchem import nwchem
from deepdrivemd.utils import Timer, parse_args

import MDAnalysis
import subprocess


class SimulationContext:
    def __init__(self, cfg: NWChemConfig):

        self.cfg = cfg
        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self._prefix = self.api.molecular_dynamics_stage.unique_name(cfg.output_path)
        self._top_file: Optional[Path] = None
        self._rst_file: Optional[Path] = None

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
    def rst_file(self) -> Optional[str]:
        if self._rst_file is None:
            return None
        return self._rst_file.as_posix()

    @property
    def reference_pdb_file(self) -> Optional[str]:
        if self.cfg.reference_pdb_file is None:
            return None
        return self.cfg.reference_pdb_file.as_posix()

    def _init_workdir(self) -> None:
        """Setup workdir and change into it."""

        self.workdir.mkdir(exist_ok=True)
        nwchem.make_nwchemrc(self.workdir,self.cfg.nwchem_top_dir)

        self._pdb_file = self._get_pdb_file()

        os.chdir(self.workdir)

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

    def _copy_rst_file(self) -> Path:
        assert self.cfg.rst_suffix is not None
        # We can abuse get_topology to get the restart file, the only difference is the suffix
        # Nevertheless, we might want to change the API.
        rst_file = self.api.get_topology(
            self.cfg.initial_pdb_dir, Path(self.pdb_file), self.cfg.rst_suffix
        )
        assert rst_file is not None
        local_rst_file = shutil.copy(rst_file, self.workdir.joinpath(rst_file.name))
        return Path(local_rst_file)

    def move_results(self) -> None:
        '''
        Move all files from the work directory to the output directory

        With NWChem this seems a bad idea as the code generates a
        number of scratch files. So this stores a lot of junk.
        '''
        if self.workdir != self.cfg.output_path:
            for p in self.workdir.iterdir():
                shutil.move(str(p), str(self.cfg.output_path.joinpath(p.name)))

class Simulation:
    def __init__(self,pdb_file):
        self.pdb_file = Path(pdb_file)
        self.reporters = []
        self.topology = app.PDBFile(str(self.pdb_file)).topology

def configure_reporters(
    sim: "app.Simulation",
    ctx: SimulationContext,
    cfg: NWChemConfig,
    report_steps: int,
    frames_per_h5: int,
) -> None:
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
            openmm_selection=cfg.nwchem_selection,
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

def configure_simulation(
            init_pdb_dir,       # init_pdb_dir=ctx.init_pdb_dir,
            pdb_file,           # pdb_file=ctx.pdb_file,
            top_file,           # top_file=ctx.top_file,
            solvent_type,       # solvent_type=cfg.solvent_type,
            dt_ps,              # dt_ps=cfg.dt_ps,
            temperature_kelvin, # temperature_kelvin=cfg.temperature_kelvin,
            nwchem_top_dir      # nwchem_top_dir=cfg.nwchem_top_dir
        ) -> None:
    # Run prepare
    nwchem.cp_ff_files(init_pdb_dir)
    nwchem.gen_input_prepare(pdb_file)
    nwchem.run_nwchem(nwchem_top_dir,"_prepare")
    # Run minimization
    nwchem.gen_input_minimize()
    nwchem.run_nwchem(nwchem_top_dir,"_minimize")
    nwchem.replace_restart_file()
    # Run equilibration (always needed as we resolvate the chemical system)
    do_dynamics = False
    time_ns = 2*dt_ps
    nwchem.gen_input_dynamics(do_dynamics,dt_ps,time_ns,temperature_kelvin,dt_ps)
    nwchem.run_nwchem(nwchem_top_dir,"_equilibrate")

def run_steps(
            dt_ps,         # dt_ps=cfg.dt_ps,
            time_ns,       # time_ns=ctx.simulation_length_ns,
            report_ps,     # report_ps=ctx.report_interval_ps,
            temperature_k, # temperature_k=cfg.temperature_kelvin,
            nwchem_top_dir # nwchem_top_dir=cfg.nwchem_top_dir
        ) -> None:
    do_dynamics = True
    nwchem.gen_input_dynamics(do_dynamics,dt_ps,time_ns,temperature_k,report_ps)
    nwchem.run_nwchem(nwchem_top_dir,"_dynamics")
    nwchem.gen_input_analysis()
    nwchem.run_nwchem(nwchem_top_dir,"_analysis")
    nwchem.fix_nwchem_xyz("nwchemdat_md.xyz")

def run_simulation(cfg: NWChemConfig) -> None:

    # Handle files
    max_retries = 3
    with Timer("molecular_dynamics_SimulationContext"):
        ctx = SimulationContext(cfg)

    # Create nwchem simulation object
    with Timer("molecular_dynamics_configure_simulation"):
        configure_simulation(
            init_pdb_dir=cfg.initial_pdb_dir.joinpath("system"),
            pdb_file=ctx.pdb_file,
            top_file=ctx.top_file,
            solvent_type=cfg.solvent_type,
            dt_ps=cfg.dt_ps,
            temperature_kelvin=cfg.temperature_kelvin,
            nwchem_top_dir=cfg.nwchem_top_dir
        )

    # openmm typed variables
    dt_ps = cfg.dt_ps * u.picoseconds
    report_interval_ps = cfg.report_interval_ps * u.picoseconds
    simulation_length_ns = cfg.simulation_length_ns * u.nanoseconds

    # Write all frames to a single HDF5 file
    # Steps between reporting DCD frames and logs
    report_steps = int(report_interval_ps / dt_ps)
    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)
    # Number of frames to report in the HDF5 file, chosen to save all reported steps
    frames_per_h5 = int(nsteps / report_steps)

    # Run simulation for nsteps
    with Timer("molecular_dynamics_step"):
        run_steps(
            dt_ps=cfg.dt_ps,
            time_ns=cfg.simulation_length_ns,
            report_ps=cfg.report_interval_ps,
            temperature_k=cfg.temperature_kelvin,
            nwchem_top_dir=cfg.nwchem_top_dir
        )

    # We need to report on structures from the trajectory file.
    # OpenMM seems to write frames DCD files, but NWChem cannot.
    # The regular OffLineReporter seems to store data in HDF5 files
    # NWChem can produce trajectory in CRD files, or XYZ files.
    # The MDAnalysis module seems to be able to read XYZ files,
    # and can write DCD files. The DCD file can be converted
    # into HDF5 using what the regular OffLineReporter could
    # do already.
    with Timer("molecular_dynamics_analysis"):
        if not ctx.reference_pdb_file:
            pdb_file = ctx.pdb_file
        else:
            pdb_file = ctx.reference_pdb_file
        sim = Simulation(pdb_file)
        sim.reporters.append(
            OfflineReporter(
                ctx.h5_prefix,
                report_steps,
                frames_per_h5=frames_per_h5,
                wrap_pdb_file=ctx.pdb_file if cfg.wrap else None,
                reference_pdb_file=ctx.reference_pdb_file,
                openmm_selection=cfg.nwchem_selection,
                mda_selection=cfg.mda_selection,
                threshold=cfg.threshold,
                contact_map=cfg.contact_map,
                point_cloud=cfg.point_cloud,
                fraction_of_contacts=cfg.fraction_of_contacts,
            )
        )
        pdb = MDAnalysis.Universe("nwchemdat_input.pdb","nwchemdat_input.pdb")
        num_frames = 0
        num_retries = 0
        while num_frames < frames_per_h5 and num_retries < max_retries:
            num_frames = 0
            trj = MDAnalysis.Universe("nwchemdat_md.pdb","nwchemdat_md.xyz")
            selection = f"bynum 1:{pdb.trajectory.n_atoms}"
            solute = trj.select_atoms(selection)
            with MDAnalysis.Writer(ctx.traj_file,pdb.trajectory.n_atoms) as wrt:
                for ts in trj.trajectory:
                    wrt.write(solute)
                    num_frames += 1
            trj.trajectory.close()
            num_retries += 1
        if num_frames < frames_per_h5 and not num_retries < max_retries:
            raise IOError("Trajectory file nwchemdat_md.xyz corrupted")
        dcd = MDAnalysis.Universe("nwchemdat_input.pdb",ctx.traj_file)
        for ts in dcd.trajectory:
            sim.reporters[0].report(sim,ts)
        # At this moment nwchemdat_md.pdb contain all atoms, i.e. solute and solvent
        # for the outlier detection we just want the solute atoms. Fix this
        # by overwriting nwchemdat_md.pdb with the input PDB file.
        subprocess.run(["cp","nwchemdat_input.pdb","nwchemdat_md.pdb"])
        # Each trajectory file is easily 700 MB in size, as we do not need this
        # data after converting the trajectory to the DCD format we should get
        # rid of this file.
        subprocess.run(["rm","nwchemdat_md.xyz"])

    # Move simulation data to persistent storage
    with Timer("molecular_dynamics_move_results"):
        if cfg.node_local_path is not None:
            ctx.move_results()


if __name__ == "__main__":
    with Timer("molecular_dynamics_stage"):
        args = parse_args()
        cfg = NWChemConfig.from_yaml(args.config)
        run_simulation(cfg)
