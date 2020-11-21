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
    local_run_dir: Path = Path("/mnt/bb/$USER/")
    # Initial data directory passed containing PDBs and optional topologies
    initial_pdb_dir: Path


class MDStageConfig(BaseSettings):
    """
    Global MD configuration (written one per experiment)
    """

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
    num_jobs: int
    # Arbitrary job parameters
    run_config: MDBaseConfig


class OpenMMConfig(MDBaseConfig):
    class MDSolvent(str, Enum):
        implicit = "implicit"
        explicit = "explicit"

    solvent_type: MDSolvent
    simulation_length_ns: float = 10
    report_interval_ps: float = 50
    wrap: bool = False
    dt_ps: float = 0.002
    temperature_kelvin: float = 310.0
    # Reference PDB file used to compute RMSD and align point cloud
    reference_pdb_file: Optional[Path]


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

    pre_exec: List[str] = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
        "conda activate /gpfs/alpine/proj-shared/med110/conda/pytorch",
        "export LANG=en_US.utf-8",
        "export LC_ALL=en_US.utf-8",
    ]
    executable: List[str] = ["/gpfs/alpine/proj-shared/med110/conda/pytorch/bin/python"]
    arguments: List[str] = ["/path/to/data/aggregation/script/aggregate.py"]
    cpu_reqs: HardwareReqs
    # Arbitrary job parameters
    run_config: AggregationBaseConfig


class BasicAggegation(AggregationBaseConfig):
    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True


class MLBaseConfig(BaseSettings):
    """Base class for specific model configs to inherit."""

    # Path to file containing preprocessed data (set by DeepDriveMD)
    input_path: Path = Path("set_by_deepdrivemd")
    # Output directory for model data (set by DeepDriveMD)
    output_path: Path = Path("set_by_deepdrivemd")
    # Model checkpoint file to load initial model weights from. Saved as .pt by CheckpointCallback.
    init_weights_path: Optional[Path]

    class Optimizer(BaseSettings):
        """ML training optimizer."""

        # PyTorch Optimizer name
        name: str = "Adam"
        # Learning rate
        lr: float = 0.0001


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
    # Retrain every i deepdrivemd iterations
    retrain_freq: int = 1
    # Arbitrary job parameters
    run_config: MLBaseConfig


class AAEModelConfig(MLBaseConfig):
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
    optimizer: MLBaseConfig.Optimizer = MLBaseConfig.Optimizer()
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
    # Project name for wandb logging
    wandb_project_name: Optional[str]


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

    pre_exec: List[str] = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh || true",
        "conda activate /gpfs/alpine/proj-shared/med110/conda/pytorch",
        "export LANG=en_US.utf-8",
        "export LC_ALL=en_US.utf-8",
        "unset CUDA_VISIBLE_DEVICES",
        "export OMP_NUM_THREADS=4",
    ]
    executable: List[str] = [
        "cat /dev/null; jsrun -n 1 -a 6 -g 6 -r 1 -c 7 /path/to/deepdrivemd/outlier_detection/optics.sh"
    ]
    arguments: List[str] = []
    cpu_reqs: HardwareReqs
    gpu_reqs: HardwareReqs
    # Arbitrary job parameters
    run_config: ODBaseConfig


class OPTICSConfig(ODBaseConfig):
    """OPTICS outlier detection algorithm configuration."""

    # TODO: move to OPTICS implementation
    # Number of outliers to detect (should be number of MD jobs + 1, incase errors)
    num_outliers: int
    # Number of points in point cloud AAE
    num_points: int
    # Inference batch size for encoder forward pass
    inference_batch_size: int = 128


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
    md_stage = MDStageConfig(
        num_jobs=10,
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
        run_config=OpenMMConfig(
            initial_pdb_dir="/path/to/initial_pdbs_and_tops",
            reference_pdb_file="/path/to/reference.pdb",
            solvent_type="explicit",
        ),
    )
    aggregation_stage = AggregationStageConfig(
        cpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=26,
            thread_type="OpenMP",
        ),
        run_config=BasicAggegation(),
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
        run_config=AAEModelConfig(),
    )
    od_stage = ODStageConfig(
        cpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=12,
            thread_type="OpenMP",
        ),
        gpu_reqs=HardwareReqs(
            processes=1,
            process_type="null",
            threads_per_process=1,
            thread_type="CUDA",
        ),
        run_config=OPTICSConfig(
            num_outliers=11,
            num_points=304,
        ),
    )

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
