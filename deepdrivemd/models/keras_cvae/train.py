import time
import json
import h5py
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from scipy.sparse import coo_matrix
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file
from deepdrivemd.selection.latest.select_model import get_model_path
from deepdrivemd.models.keras_cvae.model import conv_variational_autoencoder
from deepdrivemd.models.keras_cvae.config import KerasCVAEModelConfig


def get_init_weights(cfg: KerasCVAEModelConfig) -> Optional[str]:
    if cfg.init_weights_path is None:

        if cfg.stage_idx == 0:
            # Case for first iteration with no pretrained weights
            return

        token = get_model_path(
            stage_idx=cfg.stage_idx - 1, experiment_dir=cfg.experiment_directory
        )
        if token is None:
            # Case for no pretrained weights
            return
        else:
            # Case where model selection has run before
            _, init_weights = token
    else:
        # Case for pretrained weights
        init_weights = cfg.init_weights_path

    return init_weights.as_posix()


def get_h5_training_file(cfg: KerasCVAEModelConfig) -> Tuple[Path, List[str]]:
    # Collect training data
    api = DeepDriveMD_API(cfg.experiment_directory)
    md_data = api.get_last_n_md_runs()
    all_h5_files = md_data["data_files"]

    virtual_h5_path, h5_files = get_virtual_h5_file(
        output_path=cfg.output_path,
        all_h5_files=all_h5_files,
        last_n=cfg.last_n_h5_files,
        k_random_old=cfg.k_random_old_h5_files,
        virtual_name=f"virtual_{cfg.model_tag}",
        node_local_path=cfg.node_local_path,
    )

    return virtual_h5_path, h5_files


def sparse_to_dense(
    h5_file: Path,
    dataset_name: str,
    initial_shape: Tuple[int, ...],
    final_shape: Tuple[int, ...],
):
    """Convert sparse COO formatted contact maps to dense."""
    contact_maps = []
    with h5py.File(h5_file, "r", libver="latest", swmr=False) as f:
        for raw_indices in f[dataset_name]:
            indices = raw_indices.reshape((2, -1)).astype("int16")
            # Contact matrices are binary so we don't need to store the values
            # in HDF5 format. Instead we create a vector of 1s on the fly.
            values = np.ones(indices.shape[1]).astype("byte")
            # Construct COO formated sparse matrix
            contact_map = coo_matrix(
                (values, (indices[0], indices[1])), shape=initial_shape
            ).todense()
            contact_map = np.array(
                contact_map[: final_shape[0], : final_shape[1]], dtype=np.float16
            )
            contact_maps.append(contact_map)
    return np.array(contact_maps)


def preprocess(
    h5_file: Path,
    initial_shape: Tuple[int, ...],
    final_shape: Tuple[int, ...],
    dataset_name: str = "contact_map",
    split_pct: float = 0.8,
    shuffle: bool = True,
):

    data = sparse_to_dense(h5_file, dataset_name, initial_shape, final_shape)

    if shuffle:
        np.random.shuffle(data)

    # Split data into train and validation
    train_val_split = int(split_pct * len(data))
    train_data, valid_data = (
        data[:train_val_split],
        data[train_val_split:],
    )

    return train_data, valid_data


def main(cfg: KerasCVAEModelConfig):

    cfg.output_path.mkdir(exist_ok=True)
    init_weights = get_init_weights(cfg)
    h5_file, h5_files = get_h5_training_file(cfg)
    # Log selected H5 files
    with open(cfg.output_path / "virtual-h5-metadata.json", "w") as f:
        json.dump(h5_files, f)

    train_data, valid_data = preprocess(
        h5_file,
        cfg.initial_shape,
        cfg.final_shape,
        cfg.dataset_name,
        cfg.split_pct,
        cfg.shuffle,
    )

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
    cvae.model.summary()

    if init_weights is not None:
        cvae.model.load_weights(init_weights)

    # Optionaly train for a different number of
    # epochs on the first DDMD iterations
    if cfg.stage_idx == 0:
        epochs = cfg.initial_epochs
    else:
        epochs = cfg.epochs

    cvae.train(
        train_data,
        validation_data=valid_data,
        batch_size=cfg.batch_size,
        epochs=epochs,
    )

    # Log checkpoint
    checkpoint_path = cfg.output_path / "checkpoint"
    checkpoint_path.mkdir()
    time_stamp = time.strftime(f"epoch-{epochs}-%Y%m%d-%H%M%S.h5")
    cvae.model.save_weights(checkpoint_path / time_stamp)

    # Log loss history
    cvae.history.to_csv(cfg.output_path / "loss.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = KerasCVAEModelConfig.from_yaml(args.config)
    main(cfg)
