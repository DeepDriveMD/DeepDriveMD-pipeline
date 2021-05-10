from pathlib import Path
from typing import Tuple, Union
import numpy as np
import h5py
from scipy.sparse import coo_matrix

PathLike = Union[str, Path]


def sparse_to_dense(
    h5_file: PathLike,
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
            # Crop and reshape incase of extra 1 e.g. (N, N, 1)
            contact_map = np.array(
                contact_map[: final_shape[0], : final_shape[1]], dtype=np.float16
            ).reshape(final_shape)
            contact_maps.append(contact_map)
    return np.array(contact_maps)
