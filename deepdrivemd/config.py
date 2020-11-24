# Schema of the YAML experiment file
import json
import yaml
import argparse
from enum import Enum
from pydantic import validator
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


class MDBaseConfig(BaseSettings):
    """
    Auto-generates configuration file for run_openmm.py
    """

    # Directory to store output MD data (set by DeepDriveMD)
    result_dir: Path = Path("set_by_deepdrivemd")
    # Unique name for each MD run directory (set by DeepDriveMD)
    dir_prefix: str = "set_by_deepdrivemd"
    # PDB file used to start MD run (set by DeepDriveMD)
    pdb_file: Path = Path("set_by_deepdrivemd")
    # Node local storage path for MD run
    node_local_run_dir: Optional[Path]
    # Initial data directory passed containing PDBs and optional topologies
    initial_pdb_dir: Path = Path(".").resolve()

    @validator("initial_pdb_dir")
    def check_thread_type(cls, v):
        if not v.exists():
            raise FileNotFoundError(v.as_posix())
        if not v.is_absolute():
            raise ValueError(f"initial_pdb_dir must be an absolute path. Not {v}")
        return v


class MDStageConfig(BaseSettings):
    """
    Global MD configuration (written one per experiment)
    """

    pre_exec: List[str] = []
    executable: List[str] = []
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    gpu_reqs: GPUReqs = GPUReqs()
    num_jobs: int = 1
    # Arbitrary job parameters
    run_config: MDBaseConfig = MDBaseConfig()


class AggregationBaseConfig(BaseSettings):
    """Base class for specific aggregation configs to inherit."""

    # Path to experiment directory in order to access data API (set by DeepDriveMD)
    experiment_directory: Path = Path("set_by_deepdrivemd")
    last_n_h5_files: Optional[int]
    output_path: Path = Path("set_by_deepdrivemd")


class AggregationStageConfig(BaseSettings):
    """
    Global aggregation configuration (written one per experiment)
    """

    pre_exec: List[str] = []
    executable: List[str] = []
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    # Arbitrary job parameters
    run_config: AggregationBaseConfig = AggregationBaseConfig()


class MLBaseConfig(BaseSettings):
    """Base class for specific model configs to inherit."""

    # Path to file containing preprocessed data (set by DeepDriveMD)
    input_path: Path = Path("set_by_deepdrivemd")
    # Output directory for model data (set by DeepDriveMD)
    output_path: Path = Path("set_by_deepdrivemd")
    # Model checkpoint file to load initial model weights from. Saved as .pt by CheckpointCallback.
    init_weights_path: Optional[Path]


class MLStageConfig(BaseSettings):
    """
    Global ML configuration (written one per experiment)
    """

    pre_exec: List[str] = []
    executable: List[str] = []
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    gpu_reqs: GPUReqs = GPUReqs()
    # Retrain every i deepdrivemd iterations
    retrain_freq: int = 1
    # Arbitrary job parameters
    run_config: MLBaseConfig = MLBaseConfig()


class ODBaseConfig(BaseSettings):

    # Path to file containing preprocessed data (set by DeepDriveMD)
    input_path: Path = Path("set_by_deepdrivemd")
    # Output directory for model data (set by DeepDriveMD)
    output_path: Path = Path("set_by_deepdrivemd")
    # Model checkpoint file to load model weights for inference.
    # Saved as .pt by CheckpointCallback. (set by DeepDriveMD)
    weights_path: Path = Path("set_by_deepdrivemd")
    # Path to JSON file containing restart PDB paths (set by DeepDriveMD)
    restart_points_path: Path = Path("set_by_deepdrivemd")
    # Path to experiment directory in order to access data API (set by DeepDriveMD)
    experiment_directory: Path = Path("set_by_deepdrivemd")
    last_n_h5_files: Optional[int]


class ODStageConfig(BaseSettings):
    """
    Global outlier detection configuration (written one per experiment)
    """

    pre_exec: List[str] = []
    executable: List[str] = []
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    gpu_reqs: GPUReqs = GPUReqs()
    # Arbitrary job parameters
    run_config: ODBaseConfig = ODBaseConfig()


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
    md_stage: MDStageConfig
    aggregation_stage: AggregationStageConfig
    ml_stage: MLStageConfig
    od_stage: ODStageConfig


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
        md_stage=MDStageConfig(),
        aggregation_stage=AggregationStageConfig(),
        ml_stage=MLStageConfig(),
        od_stage=ODStageConfig(),
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
