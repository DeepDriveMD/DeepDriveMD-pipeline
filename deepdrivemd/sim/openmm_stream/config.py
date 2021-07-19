from enum import Enum
from pathlib import Path
from typing import Optional, List
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
    bp_file: Path = "md.bp"
    # adios file name copied into the simulation directory
    adios_cfg: Path = "adios.xml"
    # a template file for a simulation adios file (stream name should be replaced to be unique for each simulation)
    adios_xml_sim: Path = "adios.xml"
    # a directory with initial pdb files
    initial_pdb_dir: Path = (
        "/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_input"
    )

    @root_validator()
    def explicit_solvent_requires_top_suffix(cls, values: dict):
        top_suffix = values.get("top_suffix")
        solvent_type = values.get("solvent_type")
        if solvent_type == "explicit" and top_suffix is None:
            raise ValueError(
                "Explicit solvents require a topology file with non-None suffix"
            )
        return values


if __name__ == "__main__":
    OpenMMConfig().dump_yaml("openmm_template.yaml")
