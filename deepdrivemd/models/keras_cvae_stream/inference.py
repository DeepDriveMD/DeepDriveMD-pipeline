from pathlib import Path
from typing import Union
import numpy as np
from deepdrivemd.models.keras_cvae.config import KerasCVAEModelConfig
from deepdrivemd.models.keras_cvae.utils import sparse_to_dense
from deepdrivemd.models.keras_cvae.model import conv_variational_autoencoder

PathLike = Union[str, Path]


def generate_embeddings(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
) -> np.ndarray:

    cfg = KerasCVAEModelConfig.from_yaml(model_cfg_path)

    cvae = conv_variational_autoencoder(
        image_size=cfg.final_shape,
        channels=cfg.final_shape[-1],
        conv_layers=cfg.conv_layers,
        feature_maps=cfg.conv_filters,
        filter_shapes=cfg.conv_filter_shapes,
        strides=cfg.conv_strides,
        dense_layers=cfg.dense_layers,
        dense_neurons=cfg.dense_neurons,
        dense_dropouts=cfg.dense_dropouts,
        latent_dim=cfg.latent_dim,
    )

    cvae.model.load_weights(str(model_weights_path))

    data = sparse_to_dense(
        h5_file, cfg.dataset_name, cfg.initial_shape, cfg.final_shape
    )

    embeddings = cvae.return_embeddings(data, inference_batch_size)

    return embeddings
