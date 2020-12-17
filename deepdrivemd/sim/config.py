from enum import Enum
from pathlib import Path
from typing import Optional
from deepdrivemd.config import MDBaseConfig


class OpenMMConfig(MDBaseConfig):
    class MDSolvent(str, Enum):
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent = MDSolvent.implicit
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    wrap: bool = False
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]
    # Read outlier trajectory into memory while writing PDB file
    in_memory: bool = True


if __name__ == "__main__":
    OpenMMConfig().dump_yaml("openmm_template.yaml")
