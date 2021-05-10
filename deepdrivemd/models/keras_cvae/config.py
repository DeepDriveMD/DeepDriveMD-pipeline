from typing import List, Tuple
from deepdrivemd.config import MachineLearningTaskConfig


class KerasCVAEModelConfig(MachineLearningTaskConfig):
    # Select the n most recent HDF5 files for training
    last_n_h5_files: int = 10
    # Select k random HDF5 files to train on from previous DeepDriveMD iterations
    k_random_old_h5_files: int = 0
    # Name of the dataset in the HDF5 file.
    dataset_name: str = "contact_map"
    # Shape of contact maps stored in HDF5 file
    initial_shape: Tuple[int, ...] = (28, 28, 1)
    # Shape of contact maps passed to CVAE
    final_shape: Tuple[int, ...] = (28, 28, 1)
    # Number of epochs to train during first iteration
    initial_epochs: int = 10
    # Number of epochs to train on later iterations
    epochs: int = 10
    # Training batch size
    batch_size: int = 32
    # Percentage of data to use as training data (the rest is validation)
    split_pct: float = 0.8
    # Whether or not to shuffle training/validation data
    shuffle: bool = True

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


if __name__ == "__main__":
    KerasCVAEModelConfig().dump_yaml("keras_cvae_template.yaml")
