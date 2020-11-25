from typing import List, Tuple, Optional
import h5py


def concatenate_virtual_h5(
    input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None
):
    """
    Concatenates a list of HDF5 files containing the same
    format into a single virtual dataset.

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
