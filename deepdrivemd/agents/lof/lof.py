import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Union
import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from deepdrivemd.utils import setup_mpi_comm, setup_mpi, topk
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file
from deepdrivemd.agents.lof.config import LOFConfig
from deepdrivemd.selection.latest.select_model import get_model_path

PathLike = Union[str, Path]


def get_representation(
    model_type: str,
    model_cfg_path: PathLike,
    model_weights_path: PathLike,
    h5_file: PathLike,
    inference_batch_size: int = 128,
    device: str = "cuda:0",
    comm=None,
) -> np.ndarray:
    if model_type == "AAE3d":
        from deepdrivemd.models.aae.inference import generate_embeddings

        # Generate embeddings with a distributed forward pass
        embeddings = generate_embeddings(
            model_cfg_path,
            h5_file,
            model_weights_path,
            inference_batch_size,
            device,
            comm,
        )
    else:
        raise ValueError(f"model_type {cfg.model_type} not supported")

    return embeddings


def local_outlier_factor(
    embeddings: np.ndarray, n_outliers: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    t_start = time.time()  # Start timer
    print("Running LOF")

    # compute LOF
    clf = LocalOutlierFactor(**kwargs)
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    # Array with 1 if inlier, -1 if outlier
    clf.fit_predict(embeddings)

    # Get best scores and corresponding indices
    outlier_inds = topk(clf.negative_outlier_factor_, k=n_outliers)
    outlier_scores = clf.negative_outlier_factor_[outlier_inds]

    # Only sorts an array of size n_outliers
    sort_inds = np.argsort(outlier_scores)

    print(f"LOF Time: {time.time()- t_start}s")

    # Returns n_outlier best outliers sorted from best to worst
    return outlier_inds[sort_inds], outlier_scores[sort_inds]


def generate_outliers(
    md_data: Dict[str, List[str]],
    sampled_h5_files: List[str],
    outlier_inds: List[int],
) -> List[Dict[str, Union[str, int]]]:
    # Get all available MD data
    all_h5_files = md_data["data_files"]
    all_traj_files = md_data["traj_files"]
    all_pdb_files = md_data["structure_files"]

    # Mapping from the sampled HDF5 file to the index into md_data
    h5_sample_ind_to_all = {
        h5_file: all_h5_files.index(h5_file) for h5_file in sampled_h5_files
    }

    # Collect outlier metadata used to create PDB files down stream
    outliers = []
    for outlier_ind in outlier_inds:
        # divmod returns a tuple of quotient and remainder
        sampled_index, frame = divmod(outlier_ind, cfg.n_traj_frames)
        # Need to remap subsampled h5_file index back to all md_data
        all_index = h5_sample_ind_to_all[sampled_h5_files[sampled_index]]

        # Collect data to be passed into DeepDriveMD_API.write_task_json()
        # Data must be JSON serializable.
        outlier = {
            "structure_file": str(all_pdb_files[all_index]),
            "traj_file": str(all_traj_files[all_index]),
            "frame": int(frame),
            "outlier_ind": int(outlier_ind),
        }
        outliers.append(outlier)

    return outliers


def main(cfg: LOFConfig, distributed: bool):

    comm = setup_mpi_comm(distributed)
    comm_size, comm_rank = setup_mpi(comm)

    if comm_rank == 0:
        t_start = time.time()  # Start timer

        # Collect training data
        api = DeepDriveMD_API(cfg.experiment_directory)
        md_data = api.get_last_n_md_runs()

        virtual_h5_file, sampled_h5_files = get_virtual_h5_file(
            output_path=cfg.output_path,
            all_h5_files=md_data["data_files"],
            last_n=cfg.last_n_h5_files,
            k_random_old=cfg.k_random_old_h5_files,
            virtual_name=api.agent_stage.unique_name(cfg.output_path),
            node_local_path=cfg.node_local_path,
        )

        # Get best model hyperparameters and weights
        token = get_model_path(experiment_dir=cfg.experiment_directory)
        assert token is not None
        model_cfg_path, model_weights_path = token

    else:
        virtual_h5_file, model_cfg_path, model_weights_path = None, None, None

    if comm_size > 1:
        virtual_h5_file = comm.bcast(virtual_h5_file, 0)
        model_cfg_path = comm.bcast(model_cfg_path, 0)
        model_weights_path = comm.bcast(model_weights_path, 0)

    # Select machine learning model and generate embeddings
    embeddings = get_representation(
        cfg.model_type,
        model_cfg_path,
        model_weights_path,
        virtual_h5_file,
        cfg.inference_batch_size,
        cfg.device,
        comm,
    )

    if comm_rank == 0:
        # Perform LocalOutlierFactor outlier detection on embeddings
        outlier_inds, _ = local_outlier_factor(
            embeddings,
            cfg.num_outliers,
            n_jobs=cfg.sklearn_num_jobs,
        )

        outliers = generate_outliers(md_data, sampled_h5_files, list(outlier_inds))

        # Dump metadata to disk for MD stage
        api.agent_stage.write_task_json(outliers, cfg.stage_idx, cfg.task_idx)

        print(f"Outlier Detection Time: {time.time() - t_start}s")

    if comm is not None:
        # Final barrier
        comm.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed inference"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # set forkserver (needed for summit runs, may cause errors elsewhere)
    torch.multiprocessing.set_start_method("forkserver", force=True)

    args = parse_args()
    cfg = LOFConfig.from_yaml(args.config)
    main(cfg, args.distributed)
