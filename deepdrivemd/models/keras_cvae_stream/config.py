from pathlib import Path
from typing import List, Tuple

from deepdrivemd.config import MachineLearningTaskConfig


class KerasCVAEModelConfig(MachineLearningTaskConfig):
    # Shape of contact maps passed to CVAE
    final_shape: Tuple[int, ...] = (28, 28, 1)
    # Number of epochs
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
    # retrain the model from scratch at each iteration or start with the previously trained model
    reinit: bool = True
    use_model_checkpoint = True
    read_batch: int = 10000


if __name__ == "__main__":
    KerasCVAEModelConfig().dump_yaml("keras_cvae_template.yaml")
