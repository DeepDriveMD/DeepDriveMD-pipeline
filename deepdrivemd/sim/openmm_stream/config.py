from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from lockfile import LockFile  # type: ignore
from pydantic import root_validator

from deepdrivemd.config import MolecularDynamicsTaskConfig


class OpenMMConfig(MolecularDynamicsTaskConfig):
    class MDSolvent(str, Enum):
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent = MDSolvent.implicit
    top_suffix: Optional[str] = ".top"
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    heat_bath_friction_coef: float = 1.0
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]
    # Atom selection for openmm
    openmm_selection: List[str] = ["CA"]
    # Atom selection for MDAnalysis
    mda_selection: str = "protein and name CA"
    # Distance threshold to use for computing contact (in Angstroms)
    threshold: float = 8.0
    # Read outlier trajectory into memory while writing PDB file - not used but is in run*.py, should be cleaned out from there
    in_memory: bool = True
    # Name of bp "socket" file in simulation directory (.sst is added by adios)
    bp_file: Path = Path("md.bp")
    # adios file name copied into the simulation directory
    adios_cfg: Path = Path("adios.xml")
    # a template file for a simulation adios file (stream name should be replaced to be unique for each simulation)
    adios_xml_sim: Path = Path("adios.xml")
    # a directory with initial pdb files
    initial_pdb_dir: Path = Path()
    # should rmsd be computed or there is no reference pdb
    compute_rmsd: bool = True
    # if necessary, reduce the number of atoms participating in contact map computation to make this number divisible by:
    divisibleby: int = 2
    # directory where outliers are published
    outliers_dir: Path = Path()
    # pickle file with outliers database
    pickle_db: Path = Path()
    # probability with which velocities are copied from outlier to the new state (vs generating them randomly from a distribution with the given temperature)
    copy_velocities_p: float = 0.5
    # simulation directory
    current_dir: Path = Path()
    lock: LockFile = None

    @root_validator()
    def explicit_solvent_requires_top_suffix(
        cls, values: Dict[str, str]
    ) -> Dict[str, str]:
        top_suffix = values.get("top_suffix")
        solvent_type = values.get("solvent_type")
        if solvent_type == "explicit" and top_suffix is None:
            raise ValueError(
                "Explicit solvents require a topology file with non-None suffix"
            )
        return values


if __name__ == "__main__":
    OpenMMConfig().dump_yaml("openmm_template.yaml")
