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
    def dump_yaml(self, cfg_path):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

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


class AggregationConfig(BaseSettings):
    """
    Auto-generates configuration file
    """

    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True
    last_n_h5_files: Optional[int]

    experiment_directory: Path
    out_path: str


class AggregationStageConfig(BaseSettings):
    """
    Global aggregation configuration (written one per experiment)
    """

    pre_exec: List[str] = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
        "conda activate /gpfs/alpine/proj-shared/med110/conda/pytorch",
        "export LANG=en_US.utf-8",
        "export LC_ALL=en_US.utf-8",
    ]
    executable: List[str] = ["/gpfs/alpine/proj-shared/med110/conda/pytorch/bin/python"]
    arguments: List[str] = ["/path/to/data/aggregation/script/aggregate.py"]
    cpu_reqs: HardwareReqs

    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True
    last_n_h5_files: Optional[int]


class Optimizer(BaseSettings):
    """ML training optimizer."""

    # PyTorch Optimizer name
    name: str = "Adam"
    # Learning rate
    lr: float = 0.0001


class ModelBaseConfig(BaseSettings):
    """Base class for specific model configs to inherit."""


class MLConfig(BaseSettings):
    """Auto-generates configuration file."""

    # Path to file containing preprocessed data
    input_path: Path
    # Output directory for model data
    output_path: Path
    # Model checkpoint file to resume training. Checkpoint files saved as .pt by CheckpointCallback.
    checkpoint: Optional[Path]
    # Resume from last checkpoint
    resume: bool = False
    # Model checkpoint file to load initial model weights from. Saved as .pt by CheckpointCallback.
    init_weights_path: Optional[Path]
    # Project name for wandb logging
    wandb_project_name: Optional[str]
    # Arbitrary model parameters
    model_cfg: ModelBaseConfig


class MLStageConfig(BaseSettings):
    """
    Global ML configuration (written one per experiment)
    """

    pre_exec: List[str] = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
        "conda activate /gpfs/alpine/proj-shared/med110/conda/pytorch",
        "module load gcc/7.4.0",
        "module load cuda/10.1.243",
        "module load hdf5/1.10.4",
        "export LANG=en_US.utf-8",
        "export LC_ALL=en_US.utf-8",
        "export LD_LIBRARY_PATH=/gpfs/alpine/proj-shared/med110/atrifan/scripts/cuda/targets/ppc64le-linux/lib/:$LD_LIBRARY_PATH",
        # DDP settings
        "unset CUDA_VISIBLE_DEVICES",
        "export OMP_NUM_THREADS=4",
    ]
    executable: List[str] = [
        "cat /dev/null; jsrun -n 1 -r 1 -g 6 -a 6 -c 42 -d packed /path/to/deepdrivemd/models/aae/bin/summit.sh"
    ]
    arguments: List[str] = []
    cpu_reqs: HardwareReqs
    gpu_reqs: HardwareReqs

    # Model checkpoint file to load initial model weights from. Saved as .pt by CheckpointCallback.
    init_weights_path: Optional[Path]
    # Project name for wandb logging
    wandb_project_name: Optional[str]
    # Arbitrary model parameters
    model_cfg: ModelBaseConfig
    # Retrain every i deepdrivemd iterations
    retrain_freq: int = 1


class AAEModelConfig(ModelBaseConfig):
    # TODO: move to model implementation

    class LossWeights(BaseSettings):
        lambda_rec: float = 0.5
        lambda_gp: float = 10

    # Name of the dataset in the HDF5 file.
    dataset_name: str = "point_cloud"
    # Name of the RMSD data in the HDF5 file.
    rmsd_name: str = "rmsd"
    # Name of the fraction of contacts data in the HDF5 file.
    fnc_name: str = "fnc"
    # Model ID in for file naming
    model_id: str = "model_name"
    # Number of input points in point cloud
    num_points: int = 3375
    # Number of features per point in addition to 3D coordinates
    num_features: int = 0
    # Encoder kernel sizes
    encoder_kernel_sizes: list = [5, 5, 3, 1, 1]
    # Number of epochs to train
    epochs: int = 10
    # Training batch size
    batch_size: int = 32
    # Optimizer params
    optimizer: Optimizer = Optimizer()
    # Latent dimension of the AAE
    latent_dim: int = 64
    # Hyperparameters weighting different elements of the loss
    loss_weights: LossWeights = LossWeights()
    # Standard deviation of the prior distribution
    noise_std: float = 0.2
    # Saves embeddings every embed_interval'th epoch
    embed_interval: int = 1
    # Saves tsne plots every tsne_interval'th epoch
    tsne_interval: int = 5
    # For saving and plotting embeddings. Saves len(validation_set) / sample_interval points.
    sample_interval: int = 20
    # Number of data loaders for training
    num_data_workers: int = 0
    # String which specifies from where to feed the dataset. Valid choices are `storage` and `cpu-memory`.
    dataset_location: str = "storage"


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
    model_params: ModelBaseConfig
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
    walltime_min: int
    max_iteration: int
    cpus_per_node: int
    gpus_per_node: int
    hardware_threads_per_cpu: int
    experiment_directory: Path
    md_stage: MDStageConfig
    aggregation_stage: AggregationStageConfig
    ml_stage: MLStageConfig
    od_stage: OutlierDetectionUserConfig


def generate_sample_config():
    md_stage = MDStageConfig(
        num_jobs=10,
        initial_pdb_dir="/path/to/initial_pdbs_and_tops",
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
    aggregation_stage = AggregationStageConfig(
        cpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=26,
            thread_type="OpenMP",
        ),
    )
    ml_stage = MLStageConfig(
        cpu_reqs=HardwareReqs(
            processes=1,
            process_type="MPI",
            threads_per_process=4,
            thread_type="OpenMP",
        ),
        gpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=1,
            thread_type="CUDA",
        ),
        model_cfg=AAEModelConfig(),
    )
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
        aggregation_stage=aggregation_stage,
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
    config = generate_sample_config()
    config.dump_yaml("deepdrivemd_template.yaml")
