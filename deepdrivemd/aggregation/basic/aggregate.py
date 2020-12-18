import h5py
import argparse
import numpy as np
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.aggregation.basic.config import BasicAggegation


def concatenate_last_n_h5(cfg: BasicAggegation):

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
    fout = h5py.File(cfg.output_path, "w", libver="latest")

    # Initialize data buffers
    data = {x: [] for x in fields}

    for in_file in files:

        if cfg.verbose:
            print("Reading", in_file)

        with h5py.File(in_file, "r") as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    for field in data:
        data[field] = np.concatenate(data[field])

    # Centor of mass (CMS) subtraction
    if "point_cloud" in data:
        if cfg.verbose:
            print("Subtract center of mass (CMS) from point cloud")
        cms = np.mean(
            data["point_cloud"][:, 0:3, :].astype(np.float128), axis=2, keepdims=True
        ).astype(np.float32)
        data["point_cloud"][:, 0:3, :] -= cms

    # Create new dsets from concatenated dataset
    for field, concat_dset in data.items():
        if field == "traj_file":
            utf8_type = h5py.string_dtype("utf-8")
            fout.create_dataset("traj_file", data=concat_dset, dtype=utf8_type)
            continue

        shape = concat_dset.shape
        chunkshape = (1,) + shape[1:]
        # Create dataset
        if concat_dset.dtype != np.object:
            if np.any(np.isnan(concat_dset)):
                raise ValueError("NaN detected in concat_dset.")
            dset = fout.create_dataset(
                field, shape, chunks=chunkshape, dtype=concat_dset.dtype
            )
        else:
            dset = fout.create_dataset(
                field, shape, chunks=chunkshape, dtype=h5py.vlen_dtype(np.int16)
            )
        # write data
        dset[...] = concat_dset[...]

    # Clean up
    fout.flush()
    fout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = BasicAggegation.from_yaml(args.config)
    concatenate_last_n_h5(cfg)
