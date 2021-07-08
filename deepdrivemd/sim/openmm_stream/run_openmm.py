import shutil
import argparse
from pathlib import Path
from typing import Optional
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
from mdtools.openmm.sim import configure_simulation
from deepdrivemd.utils import Timer
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.sim.openmm.config import OpenMMConfig
from openmm_reporter import ContactMapReporter
import sys
import os
# from deepdrivemd.misc.OutlierDB import OutlierDB
from OutlierDB import *
from lockfile import LockFile
import pickle
import parmed as pmd
import numpy as np
import subprocess

class SimulationContext:
    def __init__(self, cfg: OpenMMConfig):

        self.cfg = cfg; print("cfg = ", cfg)
        self.api = DeepDriveMD_API(cfg.experiment_directory); print("api = ", self.api)
        self._prefix = self.api.molecular_dynamics_stage.unique_name(cfg.output_path); print("prefix = ", self._prefix)

        # Use node local storage if available. Otherwise, write to output directory.
        if cfg.node_local_path is not None:
            self.workdir = cfg.node_local_path.joinpath(self._prefix); print("local = ", self.workdir)
        else:
            self.workdir = cfg.output_path; print("nolocal = ", self.workdir)

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
    def reference_pdb_file(self) -> Optional[str]:
        if self.cfg.reference_pdb_file is None:
            return None
        return self.cfg.reference_pdb_file.as_posix()

    def _init_workdir(self):
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

    def move_results(self):
        if self.workdir != self.cfg.output_path:
            for p in self.workdir.iterdir():
                shutil.move(str(p), str(self.cfg.output_path.joinpath(p.name)))


def configure_reporters(
        sim: omm.app.Simulation,
        ctx: SimulationContext,
        cfg: OpenMMConfig,
        report_steps: int):
    
    sim.reporters.append(ContactMapReporter(report_steps, cfg))


def next_outlier(cfg: OpenMMConfig, sim: omm.app.Simulation):
    if(cfg.next_outlier_policy == 1):
        cfg.pickle_db = cfg.outliers_dir + "/OutlierDB.pickle"

        if(not os.path.exists(cfg.pickle_db)):
            return None

        if(cfg.lock == "set_by_deepdrivemd"):
            cfg.lock = LockFile(cfg.pickle_db)

        cfg.lock.acquire()
        with open(cfg.pickle_db, 'rb') as f:
            db = pickle.load(f)
        md5 = db.sorted_index[cfg.task_idx]
        rmsd = db.dictionary[md5][0]
        positions_pdb = cfg.outliers_dir + f"/{md5}.pdb"
        velocities_npy = cfg.outliers_dir + f"/{md5}.npy"
        shutil.copy(positions_pdb, cfg.current_dir)
        shutil.copy(velocities_npy, cfg.current_dir)
        shutil.copy(cfg.pickle_db, cfg.current_dir)
        cfg.lock.release()

        with open(cfg.current_dir + "/rmsd.txt","w") as f:
            f.write(f"{rmsd}\n")

        return positions_pdb, velocities_npy, rmsd, md5
        
    elif(cfg.next_outlier_policy == 0):
        "To implement"
        return None

def prepare_simulation(cfg: OpenMMConfig, iteration: int, sim: omm.app.Simulation):
    sim_dir = cfg.output_path/str(iteration)
    sim_dir.mkdir(exist_ok = True)
    cfg.current_dir = str(sim_dir)

    outlier = next_outlier(cfg, sim)
    if(outlier != None):
        print("There are outliers")
        positions_pdb, velocities_npy, rmsd, md5 = outlier
        while(True):
            try:
                positions = pmd.load_file(positions_pdb).positions
                velocities = np.load(velocities_npy)
                break
            except:
                print(f"Waiting for {positions_pdb} and {velocities_npy}") 
                time.sleep(5)

        sim.context.setPositions(positions)
        if(random.random() < cfg.copy_velocities_p):
            print("Copying velocities from outliers")
            sim.context.setVelocities(velocities)
        else:
            print("Generating velocities randomly")
            sim.context.setVelocitiesToTemperature(cfg.temperature_kelvin*u.kelvin, random.randint(1, 10000))
        return True
    else:
        print("There are no outliers")
        return False

def run_simulation(cfg: OpenMMConfig):

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

    report_steps = int(report_interval_ps/dt_ps)
    print("report_steps = ", report_steps)

    # Configure reporters to write output files
    with Timer("molecular_dynamics_configure_reporters"):
        configure_reporters(sim, ctx, cfg, report_steps)

    iteration = 0
    while(True):
        # Run simulation for nsteps
        print(f"Simulation iteration {iteration}"); sys.stdout.flush()
        with Timer("molecular_dynamics_step"):
            sim.step(nsteps)
        iteration += 1
        prepare_simulation(cfg, iteration, sim)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args

def adios_configuration(cfg: OpenMMConfig ):
    adios_cfg = cfg.output_path/"adios.xml"
    shutil.copy(cfg.adios_cfg, adios_cfg)
    cfg.adios_cfg = adios_cfg
    taskdir = os.path.basename(cfg.output_path)
    f = open(cfg.adios_cfg,'r')
    textxml = f.read()
    f.close()
    textxml = textxml.replace("SimulationOutput", taskdir)
    f = open(cfg.adios_cfg, 'w')
    f.write(textxml)
    f.close()

if __name__ == "__main__":
    print(subprocess.getstatusoutput("hostname")[1]); sys.stdout.flush()
    args = parse_args()
    cfg = OpenMMConfig.from_yaml(args.config)
    adios_configuration(cfg)
    cfg.bp_file = cfg.output_path/"md.bp"
    run_simulation(cfg)
