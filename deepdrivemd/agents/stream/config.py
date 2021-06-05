from typing import Optional, List, Tuple
from pydantic import root_validator, validator
from deepdrivemd.config import AgentTaskConfig


class OutlierDetectionConfig(AgentTaskConfig):
    """Outlier detection algorithm configuration."""

    # Number of outliers to detect with LOF
    num_intrinsic_outliers: int = 100
    # Number of outliers to choose from `num_intrinsic_outliers`
    # ranked by the extrinsic scoring method
    num_extrinsic_outliers: int = 100
    # Intrinsic scoring method
    intrinsic_score: Optional[str] = "lof"
    # Exrtrinsic scoring method
    extrinsic_score: Optional[str] = None
    # Number of frames in each trajectory/HDF5 file
    n_traj_frames: int = 200
    # Select the n most recent HDF5 files for outlier search
    n_most_recent_h5_files: int = 10
    # Select k random HDF5 files from previous DeepDriveMD iterations for outlier search
    k_random_old_h5_files: int = 0
    # Number of workers to use for LOF
    sklearn_num_jobs: int = -1
    # Machine learning model type
    model_type: str = "AAE3d"
    # Inference batch size for encoder forward pass
    inference_batch_size: int = 128


    agg_dir: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/aggregation_runs"
    num_agg: int = 2
    min_step_increment: int = 500
    timeout1: int = 30
    timeout2: int = 10

    best_model: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/machine_learning_runs/stage0000/task0000/published_model/best.h5"
    lastN: int = 8000

    # Model hyperparameters
    # Latent dimension of the CVAE
    latent_dim: int = 10
    # Number of convolutional layers
    conv_layers: int = 4
    # Convolutional filters
    conv_filters: List[int] = [64, 64, 64, 64]
    # Convolutional filter shapes
    conv_filter_shapes: List[Tuple[int, int]] = [(3, 3), (3, 3), (3, 3), (3, 3)]
    # Convolutional strides
    conv_strides: List[Tuple[int, int]] = [(1, 1), (2, 2), (1, 1), (1, 1)]
    # Number of dense layers
    dense_layers: int = 1
    # Number of neurons in each dense layer
    dense_neurons: List[int] = [128]
    # Dropout values for each dense layer
    dense_dropouts: List[float] = [0.25]


    best_model: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/machine_learning_runs/stage0000/task0000/published_model/best.h5"
    outlier_count: int = 120
    outlier_max: int = 4500
    outlier_min: int = 3000
    init_pdb_file: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_input/1FME-0.pdb"
    ref_pdb_file: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_reference/1FME.pdb"
    n_workers: int = 39
    init_eps: float = 1.3
    adios_xml: str = "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/aggregation/stream/adios.xml"
    batch: int = 10000


    @root_validator()
    def num_outliers_check(cls, values: dict):
        num_intrinsic_outliers = values.get("num_intrinsic_outliers")
        num_extrinsic_outliers = values.get("num_extrinsic_outliers")
        if num_extrinsic_outliers > num_intrinsic_outliers:
            raise ValueError(
                "num_extrinsic_outliers must be less than or equal to num_intrinsic_outliers"
            )
        return values

    @root_validator()
    def scoring_method_check(cls, values: dict):
        intrinsic_score = values.get("intrinsic_score")
        extrinsic_score = values.get("extrinsic_score")
        valid_intrinsic_scores = {"lof", "dbscan", None}
        valid_extrinsic_scores = {"rmsd", None}
        if intrinsic_score is None and extrinsic_score is None:
            raise ValueError("intrinsic_score and extrinsic_score cannot both be None.")
        if intrinsic_score not in valid_intrinsic_scores:
            raise ValueError(
                f"intrinsic score must be one of {valid_intrinsic_scores}, not {intrinsic_score}."
            )
        if extrinsic_score not in valid_extrinsic_scores:
            raise ValueError(
                f"extrinsic score must be one of {valid_extrinsic_scores}, not {extrinsic_score}."
            )
        return values

    @validator("model_type")
    def model_type_check(cls, v):
        valid_model_types = {"AAE3d", "keras_cvae"}
        if v not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}, not {v}.")
        return v


if __name__ == "__main__":
    OutlierDetectionConfig().dump_yaml("lof_template.yaml")
