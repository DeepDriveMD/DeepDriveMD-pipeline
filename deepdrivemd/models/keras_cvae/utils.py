from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    import numpy.typing as npt

import h5py  # type: ignore[import]
import numpy as np
from scipy.sparse import coo_matrix  # type: ignore[import]

from deepdrivemd.utils import PathLike


def sparse_to_dense(
    h5_file: PathLike,
    dataset_name: str,
    initial_shape: Tuple[int, int],
    final_shape: Union[Tuple[int, int, int], Tuple[int, int]],
) -> "npt.ArrayLike":
    """Convert sparse COO formatted contact maps to dense.

    Parameters
    ----------
    h5_file : PathLike
        The HDF5 file containing contact maps.
    dataset_name : str
        The dataset name containing the contact map indices.
    initial_shape : Tuple[int, int]
        The shape of the contact map saved in the HDF5 file.
    final_shape : Union[Tuple[int, int, int], Tuple[int, int]]
        The final shape of the contact map incase adding an extra
        dimension is necessary e.g. (D, D, 1) where D is the number
        of residues or the cropping shape.

    Returns
    -------
    npt.ArrayLike
        The output array of contact maps of shape (N, D, D) or
        (N, D, D, 1) depending on :obj:`final_shape` where N is
        the number of contact maps in the HDF5 file.
    """
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
