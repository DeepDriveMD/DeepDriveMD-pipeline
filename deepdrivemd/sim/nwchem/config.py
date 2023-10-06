from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import root_validator

from deepdrivemd.config import MolecularDynamicsTaskConfig


class NWChemConfig(MolecularDynamicsTaskConfig):
    class MDSolvent(str, Enum):
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent = MDSolvent.explicit
    top_suffix: Optional[str] = ".top" # Topology suffix
    rst_suffix: Optional[str] = ".rst" # Restart suffix
    simulation_length_ns: float = 0.02
    report_interval_ps: float = 0.02
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    #heat_bath_friction_coef: float = 1.0 # not available for Berendsen thermostat
    # Whether to wrap system, only implemented for nsp system
    # TODO: generalize this implementation.
    wrap: bool = False
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]
    # NWChem top directory (i.e. the top NWChem installation directory)
    nwchem_top_dir: Optional[Path] = None
    # Atom selection for nwchem
    nwchem_selection: List[str] = ["CA"]
    # Atom selection for MDAnalysis
    mda_selection: str = "protein and name CA"
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

    @root_validator()
    def explicit_solvent_requires_top_suffix(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        top_suffix = values.get("top_suffix")
        solvent_type = values.get("solvent_type")
        if solvent_type == "explicit" and top_suffix is None:
            raise ValueError(
                "Explicit solvents require a topology file with non-None suffix"
            )
        return values


if __name__ == "__main__":
    NWChemConfig().dump_yaml("nwchem_template.yaml")
