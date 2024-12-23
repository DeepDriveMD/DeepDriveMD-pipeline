from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import root_validator

from deepdrivemd.config import MolecularDynamicsTaskConfig


class LAMMPSConfig(MolecularDynamicsTaskConfig):
    class MDSolvent(str, Enum):
        none     = "none"
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent = MDSolvent.none
    # LAMMPS does not have a separate topology file, it just uses the input file.
    #top_suffix: Optional[str] = ".top" # Topology suffix
    # We run only short MD simulations so we have no need to restart anything.
    #rst_suffix: Optional[str] = ".rst" # Restart suffix
    simulation_length_ns: float = 0.0025
    report_interval_ps: float = 0.0025
    dt_ps: float = 0.00025
    temperature_kelvin: float = 300.0
    #heat_bath_friction_coef: float = 1.0 # not available for Berendsen thermostat
    # Whether to wrap system, only implemented for nsp system
    # TODO: generalize this implementation.
    wrap: bool = False
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]
    # LAMMPS install prefix directory
    lammps_prefix_path: Optional[Path] = None
    # Atom selection for LAMMPS
    # In some places the full atom name is used (e.g. H1 or H2) and in some places just
    # the chemical symbol survives (e.g. H). So for consistency we need to list both.
    lammps_selection: List[str] = ["H", "H1", "H2", "H3", "H4", "C", "C1", "C2", "N", "O"]
    # Atom selection for MDAnalysis
    mda_selection: str = "(name H* C* N* O*)"
    # Distance threshold to use for computing contact (in Angstroms)
    threshold: float = 8.0
    # Write contact maps to HDF5
    contact_map: bool = False
    # Write point clouds to HDF5
    point_cloud: bool = True
    # Write fraction of contacts to HDF5
    fraction_of_contacts: bool = False
    # Read outlier trajectory into memory while writing PDB file
    in_memory: bool = True
    # Directory with the initial PDB file
    initial_pdb_dir: Optional[Path] = None
    # Directory where the DeePMD models live
    train_dir: Optional[Path] = None

    #@root_validator()
    #def explicit_solvent_requires_top_suffix(
    #    cls, values: Dict[str, Any]
    #) -> Dict[str, Any]:
    #    top_suffix = values.get("top_suffix")
    #    solvent_type = values.get("solvent_type")
    #    if solvent_type == "explicit" and top_suffix is None:
    #        raise ValueError(
    #            "Explicit solvents require a topology file with non-None suffix"
    #        )
    #    return values


if __name__ == "__main__":
    LAMMPSConfig().dump_yaml("lammps_template.yaml")
