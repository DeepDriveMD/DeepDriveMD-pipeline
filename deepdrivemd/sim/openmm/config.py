from enum import Enum
from pathlib import Path
from typing import Optional, List
from deepdrivemd.config import MolecularDynamicsTaskConfig


class OpenMMConfig(MolecularDynamicsTaskConfig):
    class MDSolvent(str, Enum):
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent = MDSolvent.implicit
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0
    # Whether to wrap system, only implemented for nsp system
    # TODO: generalize this implementation.
    wrap: bool = False
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]
    # Atom selection for openmm
    openmm_selection: List[str] = ["CA"]
    # Atom selection for MDAnalysis
    mda_selection: str = "protein and name CA"
    # Distance threshold to use for computing contact (in Angstroms)
    threshold: float = 8.0
    # Write contact maps to HDF5
    contact_map: bool = True
    # Write point clouds to HDF5
    point_cloud: bool = True
    # Write fraction of contacts to HDF5
    fraction_of_contacts: bool = True
    # Read outlier trajectory into memory while writing PDB file
    in_memory: bool = True


if __name__ == "__main__":
    OpenMMConfig().dump_yaml("openmm_template.yaml")
