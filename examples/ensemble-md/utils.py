from typing import Optional
from deepdrivemd.config import BaseSettings


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
