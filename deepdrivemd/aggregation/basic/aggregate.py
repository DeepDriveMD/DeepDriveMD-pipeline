from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import numpy.typing as npt

import h5py  # type: ignore[import]
import numpy as np

from deepdrivemd.aggregation.basic.config import BasicAggegation
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.utils import parse_args


def concatenate_last_n_h5(cfg: BasicAggegation) -> None:  # noqa

    fields = []
    if cfg.rmsd:
        fields.append("rmsd")
    if cfg.fnc:
        fields.append("fnc")
    if cfg.contact_map:
        fields.append("contact_map")
    if cfg.point_cloud:
        fields.append("point_cloud")

    # Get list of input h5 files
    api = DeepDriveMD_API(cfg.experiment_directory)
    md_data = api.get_last_n_md_runs(n=cfg.last_n_h5_files)
    files = md_data["data_files"]

    if cfg.verbose:
        print(f"Collected {len(files)} h5 files.")

    # Open output file
    fout = h5py.File(cfg.output_path / "aggregate.h5", "w", libver="latest")

    # Initialize data buffers
    data: Dict[str, List["npt.ArrayLike"]] = {x: [] for x in fields}

    for in_file in files:

        if cfg.verbose:
            print("Reading", in_file)

        with h5py.File(in_file, "r") as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    concat_data: Dict[str, "npt.ArrayLike"] = {
        field: np.concatenate(data[field]) for field in data  # type: ignore[no-untyped-call]
    }
    # for field in data:
    #    data[field] = np.concatenate(data[field])  # type: ignore[no-untyped-call]

    # Centor of mass (CMS) subtraction
    if "point_cloud" in concat_data:
        if cfg.verbose:
            print("Subtract center of mass (CMS) from point cloud")
        cms = np.mean(
            concat_data["point_cloud"][:, 0:3, :].astype(np.float128),  # type: ignore[call-overload, union-attr, index]
            axis=2,
            keepdims=True,
        ).astype(np.float32)
        concat_data["point_cloud"][:, 0:3, :] -= cms  # type: ignore[call-overload, index]

    # Create new dsets from concatenated dataset
    for field, concat_dset in concat_data.items():
        if field == "traj_file":
            utf8_type = h5py.string_dtype("utf-8")
            fout.create_dataset("traj_file", data=concat_dset, dtype=utf8_type)
            continue

        shape = concat_dset.shape  # type: ignore[union-attr]
        chunkshape = (1,) + shape[1:]
        # Create dataset
        # Note: Aliases of built-in data types are now deprecated in Numpy:
        #       https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
        #       Not sure where that leaves np.float128, np.int16, etc.
        if concat_dset.dtype != object:  # type: ignore[union-attr, attr-defined]
            if np.any(np.isnan(concat_dset)):
                raise ValueError("NaN detected in concat_dset.")
            dset = fout.create_dataset(
                field,
                shape,
                chunks=chunkshape,
                dtype=concat_dset.dtype,  # type: ignore[union-attr]
            )
        else:
            dset = fout.create_dataset(
                field, shape, chunks=chunkshape, dtype=h5py.vlen_dtype(np.int16)
            )
        # write data
        dset[...] = concat_dset[...]  # type: ignore[call-overload, index]

    # Clean up
    fout.flush()
    fout.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = BasicAggegation.from_yaml(args.config)
    concatenate_last_n_h5(cfg)
