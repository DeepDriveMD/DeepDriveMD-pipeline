import random
from typing import Optional
import openmm.app as app  # type: ignore[import]
import openmm.unit as u  # type: ignore[import]
from mdtools.openmm.sim import configure_simulation  # type: ignore[import]
from deepdrivemd.config import BaseSettings
from deepdrivemd.utils import parse_args


class OpenMMSimulationParameters(BaseSettings):
    # Simulation parameters
    random_seed: int = 0
    # Can be implicit or explicit
    solvent_type: str = "implicit"
    # If simulation length is None, it will run indefinitely
    simulation_length_ns: Optional[float] = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0


class OpenMMSimulationConfig(BaseSettings):
    # Input/output files
    input_pdb_file: str
    input_top_file: Optional[str] = None
    output_traj_file: str = "traj.dcd"
    output_log_file: str = "traj.log"

    params: OpenMMSimulationParameters = OpenMMSimulationParameters()


def run_simulation(cfg: OpenMMSimulationConfig) -> None:

    # Seed for velocity initialization, handled in configure_simulation
    random.seed(cfg.params.random_seed)

    # openmm typed variables
    dt_ps = cfg.params.dt_ps * u.picoseconds
    report_interval_ps = cfg.params.report_interval_ps * u.picoseconds
    # If the simulation length is None, then we default to 10 and
    # implement logic below to run indefinitely.
    if not cfg.params.simulation_length_ns:
        simulation_length_ns = 10 * u.nanoseconds
    else:
        simulation_length_ns = cfg.params.simulation_length_ns * u.nanoseconds

    # Steps between reporting DCD frames and logs
    report_steps = int(report_interval_ps / dt_ps)
    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    sim = configure_simulation(
        pdb_file=cfg.input_pdb_file,
        top_file=cfg.input_top_file,
        solvent_type=cfg.params.solvent_type,
        gpu_index=0,
        dt_ps=cfg.params.dt_ps,
        temperature_kelvin=cfg.params.temperature_kelvin,
        heat_bath_friction_coef=cfg.params.heat_bath_friction_coef,
    )

    # Configure DCD file reporter
    sim.reporters.append(app.DCDReporter(cfg.output_traj_file, report_steps))

    # Configure simulation output log
    sim.reporters.append(
        app.StateDataReporter(
            cfg.output_log_file,
            report_steps,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )

    # If simulation_length_ns is None, then run indefinitely
    if not cfg.params.simulation_length_ns:
        while True:
            sim.step(nsteps)
    else:
        sim.step(nsteps)


if __name__ == "__main__":
    args = parse_args()
    cfg = OpenMMSimulationConfig.from_yaml(args.config)
    run_simulation(cfg)
