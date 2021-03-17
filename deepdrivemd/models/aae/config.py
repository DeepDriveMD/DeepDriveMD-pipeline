from typing import Optional, List
from deepdrivemd.config import MachineLearningTaskConfig


class AAEModelConfig(MachineLearningTaskConfig):
    # Select the n most recent HDF5 files for training
    last_n_h5_files: int = 10
    # Select k random HDF5 files to train on from previous DeepDriveMD iterations
    k_random_old_h5_files: int = 0
    # Name of the dataset in the HDF5 file.
    dataset_name: str = "point_cloud"
    # Name of the RMSD data in the HDF5 file.
    rmsd_name: str = "rmsd"
    # Name of the fraction of contacts data in the HDF5 file.
    fnc_name: str = "fnc"
    # Number of input points in point cloud
    num_points: int = 3375
    # Number of features per point in addition to 3D coordinates
    num_features: int = 0
    # Number of epochs to train during first iteration
    initial_epochs: int = 10
    # Number of epochs to train on later iterations
    epochs: int = 10
    # Training batch size
    batch_size: int = 32

    # Optimizer params
    # PyTorch Optimizer name
    optimizer_name: str = "Adam"
    # Learning rate
    optimizer_lr: float = 0.0001

    # Model hyperparameters
    # Latent dimension of the AAE
    latent_dim: int = 64
    # Encoder filter sizes
    encoder_filters: List[int] = [64, 128, 256, 256, 512]
    # Encoder kernel sizes
    encoder_kernel_sizes: List[int] = [5, 5, 3, 1, 1]
    # Generator filter sizes
    generator_filters: List[int] = [64, 128, 512, 1024]
    # Discriminator filter sizes
    discriminator_filters: List[int] = [512, 512, 128, 64]
    encoder_relu_slope: float = 0.0
    generator_relu_slope: float = 0.0
    discriminator_relu_slope: float = 0.0
    use_encoder_bias: bool = True
    use_generator_bias: bool = True
    use_discriminator_bias: bool = True
    # Mean of the prior distribution
    noise_mu: float = 0.0
    # Standard deviation of the prior distribution
    noise_std: float = 1.0
    # Hyperparameters weighting different elements of the loss
    lambda_rec: float = 0.5
    lambda_gp: float = 10

    # Training settings
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


if __name__ == "__main__":
    AAEModelConfig().dump_yaml("aae_template.yaml")
