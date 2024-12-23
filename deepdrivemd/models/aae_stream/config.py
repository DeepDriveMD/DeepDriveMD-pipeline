from pathlib import Path
from typing import List, Optional

from mdlearn.utils import BaseSettings, OptimizerConfig


class Point3dAAEConfig(BaseSettings):
    # File paths
    # Path to adois file
    input_path: Path = Path(
        "/p/gpfs1/yakushin/Outputs/305t/molecular_dynamics_runs/stage0000/task0000/0/trajectory.bp"
    )
    # Path to directory where trainer should write to (cannot already exist)
    output_path: Path = Path("TODO")
    # Optionally resume training from a checkpoint file
    resume_checkpoint: Optional[Path] = None

    # Number of points per sample. Should be smaller or equal
    # than the total number of points.
    num_points: int = 200
    # Number of additional per-point features in addition to xyz coords.
    num_features: int = 0
    # Name of scalar datasets.
    scalar_dset_names: List[str] = []
    # If True, subtract center of mass from batch and shift and scale
    # batch by the full dataset statistics.
    cms_transform: bool = True
    # Sets requires_grad torch.Tensor parameter for scalars specified
    # by scalar_dset_names. Set to True, to use scalars for multi-task
    # learning. If scalars are only required for plotting, then set it as False.
    scalar_requires_grad: bool = False
    # Percentage of data to be used as training data after a random split.
    split_pct: float = 0.8
    # Random seed for shuffling train/validation data
    seed: int = 333
    # Whether or not to shuffle train/validation data
    shuffle: bool = True
    # Number of epochs to train
    epochs: int = 30
    # Training batch size
    batch_size: int = 32
    # Pretrained model weights
    init_weights: Optional[str] = ""
    # AE (encoder/decoder) optimizer params
    ae_optimizer: OptimizerConfig = OptimizerConfig(name="Adam", hparams={"learning_rate": 0.0001})
    # Discriminator optimizer params
    disc_optimizer: OptimizerConfig = OptimizerConfig(
        name="Adam", hparams={"learning_rate": 0.0001}
    )

    # Model hyperparameters
    latent_dim: int = 16
    encoder_bias: bool = True
    encoder_relu_slope: float = 0.0
    encoder_filters: List[int] = [64, 128, 256, 256, 512]
    encoder_kernels: List[int] = [5, 5, 3, 1, 1]
    decoder_bias: bool = True
    decoder_relu_slope: float = 0.0
    decoder_affine_widths: List[int] = [64, 128, 512, 1024]
    discriminator_bias: bool = True
    discriminator_relu_slope: float = 0.0
    discriminator_affine_widths: List[int] = [512, 128, 64]
    # Mean of the prior distribution
    noise_mu: float = 0.0
    # Standard deviation of the prior distribution
    noise_std: float = 1.0
    # Releative weight to put on gradient penalty
    lambda_gp: float = 10.0
    # Releative weight to put on reconstruction loss
    lambda_rec: float = 0.5

    # Training settings
    # Number of data loaders for training
    num_data_workers: int = 16
    # Number of samples loaded in advance by each worker
    prefetch_factor: int = 2
    # Log checkpoint file every `checkpoint_log_every` epochs
    # checkpoint_log_every: int = 1
    # Log latent space plot `plot_log_every` epochs
    # plot_log_every: int = 1

    # Validation settings
    # Method used to visualize latent space
    # plot_method: str = "TSNE"
    # Number of validation samples to run visualization on
    # plot_n_samples: Optional[int] = None

    # minimum number of steps in each aggregated file before the model is trained
    min_step_increment: int = 5000
    # take up to this number of samples from each aggregated file to train the model
    max_steps: int = 8000
    # if the loss is greater than this, do not publish the model, retrain the model from scratch at next iteration regardless of reinit value
    max_loss: int = 10000
    # number of aggregators
    num_agg: int = 12
    # if num_agg adios aggregated files are not available, sleep for timeout1 before trying again
    timeout1: int = 30
    # if less than min_step_increment is available in each aggregated file, sleep for timeout2 before trying again
    timeout2: int = 10
    # directory with aggregated tasks subdirectories
    agg_dir: Path = Path()
    # where to publish a trained model for the outlier search to pick up
    published_model_dir: Path
    # temporary directory with model checkpoints
    checkpoint_dir: Path
    # adios xml configuration file for aggregators
    adios_xml_agg: Path
    adios_xml_agg_4ml: Path
    # retrain the model from scratch at each iteration or start with the previously trained model
    reinit: bool = False
    use_model_checkpoint = True
    read_batch: int = 10000

    experiment_directory: Path
    init_weights_path: Path
    model: str = "aae"
    model_tag: str = "aae"
    node_local_path: Path = Path("/tmp")
    stage_idx: int = 0
    task_idx: int = 0
