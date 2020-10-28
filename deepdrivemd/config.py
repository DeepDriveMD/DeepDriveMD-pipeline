# Schema of the YAML experiment file
import json
import yaml
import argparse
from enum import Enum
from pydantic import BaseSettings as _BaseSettings
from pathlib import Path
from typing import Optional, List, Union
from typing import TypeVar, Type

_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    def dump_yaml(self, file):
        yaml.dump(json.loads(self.json()), file, indent=4)

    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class HardwareReqs(BaseSettings):
    processes: int
    process_type: str
    threads_per_process: int
    thread_type: str


class MDSolvent(str, Enum):
    implicit = "implicit"
    explicit = "explicit"


class MDConfig(BaseSettings):
    """
    Auto-generates configuration file for run_openmm.py
    """

    pdb_file: Path
    initial_pdb_dir: Path
    reference_pdb_file: Optional[Path]
    local_run_dir: Path = Path("/mnt/bb/$USER/")
    solvent_type: MDSolvent
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    wrap: bool = False
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    result_dir: Path
    h5_cp_path: Path
    omm_dir_prefix: str


class MDStageConfig(BaseSettings):
    """
    Global MD configuration (written one per experiment)
    """

    num_jobs: int
    initial_pdb_dir: Path
    reference_pdb_file: Optional[Path]
    solvent_type: MDSolvent
    temperature_kelvin: float = 310.0
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    wrap: bool = False
    local_run_dir: Path = Path("/mnt/bb/$USER/")
    pre_exec: List[str] = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
        "module load cuda/9.1.85",
        "conda activate /gpfs/alpine/proj-shared/med110/conda/openmm",
        "export PYTHONPATH=/path/to/run_openmm/directory:$PYTHONPATH",
    ]
    executable: List[str] = ["/gpfs/alpine/proj-shared/med110/conda/openmm/bin/python"]
    arguments: List[str] = ["/path/to/run_openmm/run_openmm.py"]
    cpu_reqs: HardwareReqs
    gpu_reqs: HardwareReqs


class AAEModelConfig(BaseSettings):
    fraction: float = 0.2
    last_n_files: int = 1
    last_n_files_eval: int = 1
    batch_size: int = 1
    input_shape: List[int] = [1, 32, 32]
    itemsize: int = 1
    mixed_precision: bool = True

    # Model params
    enc_conv_kernels: List[int] = [5, 5, 5, 5]
    # Encoder filters define OUTPUT filters per layer
    enc_conv_filters: List[int] = [100, 100, 100, 100]  # 64, 64, 64, 32
    enc_conv_strides: List[int] = [1, 1, 2, 1]
    dec_conv_kernels: List[int] = [5, 5, 5, 5]
    # Decoder filters define INPUT filters per layer
    dec_conv_filters: List[int] = [100, 100, 100, 100]
    dec_conv_strides: List[int] = [1, 2, 1, 1]
    dense_units: int = 64  # 128
    latent_ndim: int = 10  # 3
    mixed_precision: bool = True
    activation: str = "relu"
    # Setting full_precision_loss to False as we do not support it yet.
    full_precision_loss: bool = False
    reconstruction_loss_reduction_type: str = "sum"
    kl_loss_reduction_type: str = "sum"
    model_random_seed: Optional[int] = None
    data_random_seed: Optional[int] = None

    # Optimizer params
    epsilon: float = 1.0e-8
    beta1: float = 0.2
    beta2: float = 0.9
    decay: float = 0.9
    momentum: float = 0.9
    optimizer_name: str = "rmsprop"
    allowed_optimizers: List[str] = ["sgd", "sgdm", "adam", "rmsprop"]
    learning_rate: float = 2.0e-5
    loss_scale: float = 1


class ExtrinsicScore(str, Enum):
    none = "none"
    rmsd_to_reference_state = "rmsd_to_reference_state"


class OutlierDetectionUserConfig(BaseSettings):
    num_outliers: int = 500
    extrinsic_outlier_score: ExtrinsicScore = ExtrinsicScore.none
    sklearn_num_cpus: int = 16
    local_scratch_dir: Path = Path("/raid/scratch")
    max_num_old_h5_files: int = 1000
    # Run parameters
    run_command: str = ""
    environ_setup: List[str] = []
    num_nodes: int = 1
    ranks_per_node: int = 1
    gpus_per_node: int = 8


class OutlierDetectionRunConfig(OutlierDetectionUserConfig):
    outlier_results_dir: Path
    model_params: AAEModelConfig
    md_dir: Path
    model_weights_dir: Path
    walltime_min: int
    outlier_predict_batch_size: int


class ExperimentConfig(BaseSettings):
    """
    Master configuration
    """

    title: str
    resource: str
    queue: str
    schema_: str
    project: str
    cpus_per_node: int
    gpus_per_node: int
    hardware_threads_per_cpu: int
    max_iteration: int
    experiment_directory: Path
    walltime_min: int
    md_stage: MDStageConfig
    ml_stage: AAEModelConfig
    od_stage: OutlierDetectionUserConfig


def generate_sample_config():
    md_stage = MDStageConfig(
        num_jobs=10,
        initial_configs_dir="/path/to/initial_pdbs_and_tops",
        reference_pdb_file="/path/to/reference.pdb",
        solvent_type="explicit",
        cpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=4,
            thread_type="OpenMP",
        ),
        gpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=1,
            thread_type="CUDA",
        ),
    )
    ml_stage = AAEModelConfig()
    od_stage = OutlierDetectionUserConfig()

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
        md_stage=md_stage,
        ml_stage=ml_stage,
        od_stage=od_stage,
    )


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


if __name__ == "__main__":
    with open("deepdrivemd_template.yaml", "w") as fp:
        config = generate_sample_config()
        config.dump_yaml(fp)
