from pathlib import Path
from typing import List, Tuple

from deepdrivemd.config import AgentTaskConfig


class OutlierDetectionConfig(AgentTaskConfig):
    """Outlier detection algorithm configuration."""

    # top aggregation directory
    agg_dir: Path = Path()

    # number of aggregators
    num_agg: int = 2
    # minimum acceptable number of steps in each aggregation file
    min_step_increment: int = 500
    # sleep for timeout1 seconds if less than num_agg adios files are available
    timeout1: int = 30
    # sleep for timeout2 seconds if less than min_step_increment number of steps is available in each aggregated file; same timeout2 is used to wait for the model to become available
    timeout2: int = 10
    # path to the best model
    best_model: Path = Path()
    # use up to lastN last steps from each aggregated file to cluster and find outliers
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

    # number of attempts to find between outlier_min and outlier_max
    outlier_count: int = 120
    # maximum number of outliers
    outlier_max: int = 4500
    # minimum number of outliers
    outlier_min: int = 3000
    # path to the intial pdb file
    init_pdb_file: Path = Path()

    # path to the reference pdb file
    ref_pdb_file: Path = Path()

    # initial value for eps for dbscan
    init_eps: float = 1.3
    # initial value for min_samples for dbscan
    init_min_samples: int = 10
    # adios xml configuration file for aggregated data
    adios_xml_agg: Path = ""
    # batch file for reading data from adios file
    read_batch: int = 10000
    # use rapids version of TSNE or scikit-learn version in postproduction when computing embeddings
    project_gpu: bool = False
    # use project_lastN last samples from each aggregator to search for outliers
    project_lastN: int = 8000
    num_sim: int = 120
    use_outliers: bool = True
    use_random_outliers: bool = False
    compute_rmsd: bool = True
    compute_zcentroid: bool = False
    final_shape: List[int] = [28, 28, 1]
    outlier_selection: str = "rmsd"
    multi_ligand_table: Path = Path()
    model: str = "cvae"
    num_points: int = 539
    num_features: int = 0
    """
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
    """


if __name__ == "__main__":
    OutlierDetectionConfig().dump_yaml("dbscan_template.yaml")
