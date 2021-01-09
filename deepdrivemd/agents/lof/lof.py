import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Union
import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from deepdrivemd.utils import setup_mpi_comm, setup_mpi, bestk
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file, parse_h5
from deepdrivemd.agents.lof.config import OutlierDetectionConfig
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


def get_intrinsic_score(
    embeddings: np.ndarray, cfg: OutlierDetectionConfig
) -> Tuple[np.ndarray, np.ndarray]:

    if cfg.intrinsic_score == "lof":
        # Perform LocalOutlierFactor outlier detection on embeddings
        t_start = time.time()  # Start timer
        print("Running LOF")

        # compute LOF
        clf = LocalOutlierFactor(n_jobs=cfg.sklearn_num_jobs)
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        # Array with 1 if inlier, -1 if outlier
        clf.fit_predict(embeddings)

        assert cfg.num_intrinsic_outliers is not None
        # Get best scores and corresponding indices
        intrinsic_scores, intrinsic_inds = bestk(
            clf.negative_outlier_factor_, k=cfg.num_intrinsic_outliers
        )
        print(f"LOF Time: {time.time()- t_start}s")
    else:
        # If no intrinsic_score, simply return all the data
        intrinsic_inds = np.arange(len(embeddings))
        intrinsic_scores = np.zeros(len(embeddings))

    # Returns n_outlier best outliers sorted from best to worst
    return intrinsic_scores, intrinsic_inds


def get_extrinsic_score(
    intrinsic_inds: np.ndarray, virtual_h5_file: Path, cfg: OutlierDetectionConfig
) -> Tuple[np.ndarray, np.ndarray]:

    t_start = time.time()  # Start timer

    if cfg.extrinsic_score == "rmsd":
        # Get all RMSD values from virutal HDF5 file
        rmsds = parse_h5(virtual_h5_file, fields=["rmsd"])["rmsd"]
        # Select the subset choosen with the intrinsic score method
        intrinsic_rmsds = rmsds[intrinsic_inds]
        # Find the best points within the selected subset
        extrinsic_scores, extrinsic_inds = bestk(
            intrinsic_rmsds, k=cfg.num_extrinsic_outliers
        )
    else:
        # If no extrinsic_score, simply return the intrinsic selection
        extrinsic_inds = np.arange(cfg.num_extrinsic_outliers)
        extrinsic_scores = np.zeros(len(extrinsic_inds))

    print(f"get_extrinsic_score time: {time.time()- t_start}s")

    return extrinsic_scores, extrinsic_inds


def generate_outliers(
    md_data: Dict[str, List[str]],
    sampled_h5_files: List[str],
    outlier_inds: List[int],
    intrinsic_scores: List[float],
    extrinsic_scores: List[float],
) -> List[Dict[str, Union[str, int]]]:

    assert len(intrinsic_scores) == len(extrinsic_scores)
    assert len(intrinsic_scores) == len(outlier_inds)

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
    for outlier_ind, intrinsic_score, extrinsic_score in zip(
        outlier_inds, intrinsic_scores, extrinsic_scores
    ):
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
            "intrinsic_score": float(intrinsic_score),
            "extrinsic_score": float(extrinsic_score),
        }
        outliers.append(outlier)

    return outliers


def main(cfg: OutlierDetectionConfig, encoder_gpu: int, distributed: bool):

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
            last_n=cfg.n_most_recent_h5_files,
            k_random_old=cfg.k_random_old_h5_files,
            virtual_name=api.agent_stage.unique_name(cfg.output_path),
            node_local_path=cfg.node_local_path,
        )
        with open(cfg.output_path.joinpath("virtual-h5-metadata.json"), "w") as f:
            json.dump(sampled_h5_files, f)

        # Get best model hyperparameters and weights
        token = get_model_path(api=api)
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
        encoder_gpu,
        comm,
    )

    if comm_rank == 0:

        intrinsic_scores, intrinsic_inds = get_intrinsic_score(embeddings, cfg)

        # Prune the best intrinsically ranked points with an extrinsic score
        extrinsic_scores, extrinsic_inds = get_extrinsic_score(
            intrinsic_inds, virtual_h5_file, cfg
        )

        # Take the subset of indices selected by the extrinsic method
        pruned_intrinsic_scores = intrinsic_scores[extrinsic_inds]
        pruned_intrinsic_inds = intrinsic_inds[extrinsic_inds]

        outliers = generate_outliers(
            md_data,
            sampled_h5_files,
            pruned_intrinsic_inds,
            pruned_intrinsic_scores,
            extrinsic_scores,
        )

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
        "-E", "--encoder_gpu", help="GPU to place encoder", type=int, default=0
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
    cfg = OutlierDetectionConfig.from_yaml(args.config)
    main(cfg, args.encoder_gpu, args.distributed)
