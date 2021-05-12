"""Data utility functions for handling HDF5 files."""

import h5py
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

PathLike = Union[str, Path]


def concatenate_virtual_h5(
    input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None
):
    r"""Concatenate HDF5 files into a virtual HDF5 file.

    Concatenates a list `input_file_names` of HDF5 files containing
    the same format into a single virtual dataset.

    Parameters
    ----------
    input_file_names : List[str]
        List of HDF5 file names to concatenate.
    output_name : str
        Name of output virtual HDF5 file.
    fields : Optional[List[str]]
        Which dataset fields to concatenate. Will concatenate all fields by default.
    """

    # Open first file to get dataset shape and dtype
    # Assumes uniform number of data points per file
    h5_file = h5py.File(input_file_names[0], "r")

    if not fields:
        fields = list(h5_file.keys())

    # Helper function to output concatenated shape
    def concat_shape(shape: Tuple[int]) -> Tuple[int]:
        return (len(input_file_names) * shape[0], *shape[1:])

    # Create a virtual layout for each input field
    layouts = {
        field: h5py.VirtualLayout(
            shape=concat_shape(h5_file[field].shape),
            dtype=h5_file[field].dtype,
        )
        for field in fields
    }

    with h5py.File(output_name, "w", libver="latest") as f:
        for field in fields:
            for i, filename in enumerate(input_file_names):
                shape = h5_file[field].shape
                vsource = h5py.VirtualSource(filename, field, shape=shape)
                layouts[field][i * shape[0] : (i + 1) * shape[0], ...] = vsource

            f.create_virtual_dataset(field, layouts[field])

    h5_file.close()


def get_virtual_h5_file(
    output_path: Path,
    all_h5_files: List[str],
    last_n: int = 0,
    k_random_old: int = 0,
    virtual_name: str = "virtual",
    node_local_path: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    r"""Create and return a virtual HDF5 file.

    Create a virtual HDF5 file from the `last_n` files
    in `all_h5_files` and a random selection of `k_random_old`.

    Parameters
    ----------
    output_path : Path
        Directory to write virtual HDF5 file to.
    all_h5_files : List[str]
        List of HDF5 files to select from.
    last_n : int, optional
        Chooses the last n files in `all_h5_files` to concatenate
        into a virtual HDF5 file. Defaults to all the files.
    k_random_old : int
        Chooses k random files not in the `last_n` files to
        concatenate into the virtual HDF5 file. Defaults to
        choosing no random old files.
    virtual_name : str
        The name of the virtual HDF5 file to be written
        e.g. `virtual_name == virtual` implies the file will
        be written to `output_path/virtual.h5`.
    node_local_path : Optional[Path]
        An optional path to write the virtual file to that could
        be a node local storage. Will also copy all selected HDF5
        files in `all_h5_files` to the same directory.

    Returns
    -------
    Path
        The path to the created virtual HDF5 file.
    List[str]
        The selected HDF5 files from `last_n` and `k_random_old`
        used to make the virtual HDF5 file.

    Raises
    ------
    ValueError
        If `all_h5_files` is empty.
        If `last_n` is greater than len(all_h5_files).
    """

    if not all_h5_files:
        raise ValueError("Tried to create virtual HDF5 file from empty all_h5_files")
    if len(all_h5_files) < last_n:
        raise ValueError("last_n is greater than the number files in all_h5_files")

    # Partition all HDF5 files into old and new
    last_n_h5_files = all_h5_files[-1 * last_n :]
    old_h5_files = all_h5_files[: -1 * last_n]

    # Get a random sample of old HDF5 files, or use all
    # if the length of old files is less then k_random_old
    if len(old_h5_files) > k_random_old:
        old_h5_files = random.sample(old_h5_files, k=k_random_old)

    # Combine all new files and some old files
    h5_files = old_h5_files + last_n_h5_files

    # Always make a virtual file in long term storage
    virtual_h5_file = output_path.joinpath(f"{virtual_name}.h5")
    concatenate_virtual_h5(h5_files, virtual_h5_file.as_posix())

    # If node local storage optimization is available, then
    # copy all HDF5 files to node local storage and make a
    # separate virtual HDF5 file on node local storage.
    if node_local_path is not None:
        tmp_h5_files = [shutil.copy(f, node_local_path) for f in h5_files]
        virtual_h5_file = node_local_path.joinpath(f"{virtual_name}.h5")
        concatenate_virtual_h5(tmp_h5_files, virtual_h5_file.as_posix())

    # Returns node local virtual file if available
    return virtual_h5_file, h5_files


def parse_h5(path: PathLike, fields: List[str]) -> Dict[str, np.ndarray]:
    r"""Helper function for accessing data fields in a HDF5 file.

    Parameters
    ----------
    path : Union[Path, str]
        Path to HDF5 file.
    fields : List[str]
        List of dataset field names inside of the HDF5 file.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary maping each field name in `fields` to a numpy
        array containing the data from the associated HDF5 dataset.
    """
    data = {}
    with h5py.File(path, "r") as f:
        for field in fields:
            data[field] = f[field][...]
    return data
