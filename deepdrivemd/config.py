# Schema of the YAML experiment file
import json
import yaml
import argparse
from enum import Enum
from pydantic import validator, root_validator
from pydantic import BaseSettings as _BaseSettings
from pathlib import Path
from typing import Optional, List, Union
from typing import TypeVar, Type

_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class CPUReqs(BaseSettings):
    """radical.entk task.cpu_reqs parameters."""

    class CPUProcessType(str, Enum):
        mpi = "MPI"

    class CPUThreadType(str, Enum):
        open_mp = "OpenMP"

    processes: int = 1
    process_type: Optional[CPUProcessType]
    threads_per_process: int = 1
    thread_type: Optional[CPUThreadType]


class GPUReqs(BaseSettings):
    """radical.entk task.gpu_reqs parameters."""

    class GPUProcessType(str, Enum):
        mpi = "MPI"

    class GPUThreadType(str, Enum):
        open_mp = "OpenMP"
        cuda = "CUDA"

    processes: int = 0
    process_type: Optional[GPUProcessType]
    threads_per_process: int = 0
    thread_type: Optional[GPUThreadType]


class BaseTaskConfig(BaseSettings):
    """Base configuration for all TaskConfig objects."""

    # Path to experiment directory in order to access data API (set by DeepDriveMD)
    experiment_directory: Path = Path("set_by_deepdrivemd")
    # Unique stage index (set by DeepDriveMD)
    stage_idx: int = 0
    # Unique task index (set by DeepDriveMD)
    task_idx: int = 0
    # Output directory for model data (set by DeepDriveMD)
    output_path: Path = Path("set_by_deepdrivemd")
    # Node local storage path
    node_local_path: Optional[Path] = Path("set_by_deepdrivemd")


class BaseStageConfig(BaseSettings):
    """Base configuration for all StageConfig objects."""

    pre_exec: List[str] = []
    executable: List[str] = []
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    gpu_reqs: GPUReqs = GPUReqs()


class MolecularDynamicsTaskConfig(BaseTaskConfig):
    """
    Auto-generates configuration file for run_openmm.py
    """

    # PDB file used to start MD run (set by DeepDriveMD)
    pdb_file: Optional[Path] = Path("set_by_deepdrivemd")
    # Initial data directory passed containing PDBs and optional topologies
    initial_pdb_dir: Path = Path(".").resolve()

    @validator("initial_pdb_dir")
    def initial_pdb_dir_must_exist_with_valid_pdbs(cls, v):
        if not v.exists():
            raise FileNotFoundError(v.as_posix())
        if not v.is_absolute():
            raise ValueError(f"initial_pdb_dir must be an absolute path. Not {v}")
        if any("__" in p.as_posix() for p in v.glob("*/*.pdb")):
            raise ValueError("Initial PDB files cannot contain a double underscore __")
        return v

    @root_validator
    def pdb_file_and_restart_point_both_not_none(cls, values):
        restart_point = values.get("restart_point")
        pdb_file = values.get("pdb_file")
        if restart_point is None and pdb_file is None:
            raise ValueError("pdb_file and restart_point cannot both be None")
        return values


class MolecularDynamicsStageConfig(BaseStageConfig):
    """
    Global MD configuration (written one per experiment)
    """

    num_tasks: int = 1
    # Arbitrary task parameters
    task_config: MolecularDynamicsTaskConfig = MolecularDynamicsTaskConfig()


class AggregationTaskConfig(BaseTaskConfig):
    """Base class for specific aggregation configs to inherit."""


class AggregationStageConfig(BaseStageConfig):
    """
    Global aggregation configuration (written one per experiment)
    """

    # Whether or not to skip aggregation stage
    skip_aggregation: bool = False
    # Arbitrary task parameters
    task_config: AggregationTaskConfig = AggregationTaskConfig()


class MachineLearningTaskConfig(BaseTaskConfig):
    """Base class for specific model configs to inherit."""

    # Model ID for file naming (set by DeepDriveMD)
    model_tag: str = "set_by_deepdrivemd"
    # Model checkpoint file to load initial model weights from. Saved as .pt by CheckpointCallback.
    init_weights_path: Optional[Path]


class MachineLearningStageConfig(BaseStageConfig):
    """
    Global ML configuration (written one per experiment)
    """

    # Retrain every i deepdrivemd iterations
    retrain_freq: int = 1
    # Arbitrary task parameters
    task_config: MachineLearningTaskConfig = MachineLearningTaskConfig()


class ModelSelectionTaskConfig(BaseTaskConfig):
    """Base class for specific model selection configs to inherit."""


class ModelSelectionStageConfig(BaseStageConfig):
    """
    Global ML configuration (written one per experiment)
    """

    # Arbitrary task parameters
    task_config: ModelSelectionTaskConfig = ModelSelectionTaskConfig()


class AgentTaskConfig(BaseTaskConfig):
    """Base class for specific agent configs to inherit."""


class AgentStageConfig(BaseStageConfig):
    """
    Global agent configuration (written one per experiment)
    """

    # Arbitrary job parameters
    task_config: AgentTaskConfig = AgentTaskConfig()


class ExperimentConfig(BaseSettings):
    """
    Master configuration
    """

    title: str
    resource: str
    queue: str
    schema_: str
    project: str
    walltime_min: int
    max_iteration: int
    cpus_per_node: int
    gpus_per_node: int
    hardware_threads_per_cpu: int
    experiment_directory: Path
    node_local_path: Optional[Path]
    molecular_dynamics_stage: MolecularDynamicsStageConfig
    aggregation_stage: AggregationStageConfig
    machine_learning_stage: MachineLearningStageConfig
    model_selection_stage: ModelSelectionStageConfig
    agent_stage: AgentStageConfig

    @validator("experiment_directory")
    def experiment_directory_cannot_exist(cls, v):
        if v.exists():
            raise FileNotFoundError(f"experiment_directory already exists! {v}")
        if not v.is_absolute():
            raise ValueError(f"experiment_directory must be an absolute path! Not {v}")
        return v


def generate_sample_config():
    return ExperimentConfig(
        title="COVID-19 - Workflow2",
        resource="ornl.summit",
        queue="batch",
        schema_="local",
        project="MED110",
        walltime_min=360,
        cpus_per_node=42,
        hardware_threads_per_cpu=4,
        gpus_per_node=6,
        max_iteration=4,
        experiment_directory="/path/to/experiment",
        node_local_path=None,
        molecular_dynamics_stage=MolecularDynamicsStageConfig(),
        aggregation_stage=AggregationStageConfig(),
        machine_learning_stage=MachineLearningStageConfig(),
        model_selection_stage=ModelSelectionStageConfig(),
        agent_stage=AgentStageConfig(),
    )


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


if __name__ == "__main__":
    config = generate_sample_config()
    config.dump_yaml("deepdrivemd_template.yaml")
