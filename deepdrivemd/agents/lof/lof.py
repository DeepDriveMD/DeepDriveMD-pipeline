import argparse
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt

import numpy as np
from sklearn.neighbors import LocalOutlierFactor  # type: ignore[import]

from deepdrivemd.agents.lof.config import OutlierDetectionConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file, parse_h5
from deepdrivemd.selection.latest.select_model import get_model_path
from deepdrivemd.utils import PathLike, Timer, bestk, setup_mpi, setup_mpi_comm


def get_representation(
    model_type: str,
    model_cfg_path: PathLike,
    model_weights_path: PathLike,
    h5_file: PathLike,
    inference_batch_size: int = 128,
    gpu_id: int = 0,
    comm: Optional[Any] = None,
) -> "npt.ArrayLike":
    if model_type == "AAE3d":
        from deepdrivemd.models.aae.inference import generate_embeddings

        # Generate embeddings with a distributed forward pass
        embeddings = generate_embeddings(
            model_cfg_path,
            h5_file,
            model_weights_path,
            inference_batch_size,
            gpu_id,
            comm,
        )
    elif model_type == "keras_cvae":
        from deepdrivemd.models.keras_cvae.inference import generate_embeddings  # type: ignore[no-redef]

        embeddings = generate_embeddings(  # type: ignore[call-arg]
            model_cfg_path,
            h5_file,
            model_weights_path,
            inference_batch_size,
        )
    else:
        raise ValueError(f"model_type {cfg.model_type} not supported")

    return embeddings


def run_dbscan(data: "npt.ArrayLike", eps: float = 0.35) -> "npt.ArrayLike":
    # RAPIDS.ai import as needed
    import cupy as cp  # type: ignore[import]
    from cuml import DBSCAN as DBSCAN  # type: ignore[import]

    db = DBSCAN(eps=eps, min_samples=10, max_mbytes_per_batch=500).fit(cp.asarray(data))
    outlier_inds = np.flatnonzero(db.labels_.to_array() == -1)
    return outlier_inds


def dbscan_outlier_search(
    embeddings: "npt.ArrayLike", num_intrinsic_outliers: int
) -> "npt.ArrayLike":
    """Find best eps and return corresponding outlier indices."""

    # TODO: Move these parameters to config
    eps = 1.3
    outlier_min = num_intrinsic_outliers
    outlier_max = num_intrinsic_outliers + 2000
    attempts = 120

    for _ in range(attempts):
        n_outlier = 0
        try:
            outliers = run_dbscan(embeddings, eps=eps)
            n_outlier = len(outliers)  # type: ignore[arg-type]
        except Exception as e:
            print(e, "\nNo outliers found")

        if n_outlier > outlier_max:
            eps += 0.09 * random.random()
        elif n_outlier < outlier_min:
            eps = max(0.01, eps - 0.09 * random.random())
        else:
            return outliers

    raise ValueError("Found no outliers after DBSCAN search.")


def get_intrinsic_score(
    embeddings: "npt.ArrayLike", cfg: OutlierDetectionConfig
) -> Tuple["npt.ArrayLike", "npt.ArrayLike"]:

    if cfg.intrinsic_score == "lof":
        # Perform LocalOutlierFactor outlier detection on embeddings
        clf = LocalOutlierFactor(n_jobs=cfg.sklearn_num_jobs)
        embeddings = np.nan_to_num(embeddings, nan=0.0)  # type: ignore[no-untyped-call]
        # Array with 1 if inlier, -1 if outlier
        clf.fit_predict(embeddings)

        assert cfg.num_intrinsic_outliers is not None
        # Get best scores and corresponding indices
        intrinsic_scores, intrinsic_inds = bestk(
            clf.negative_outlier_factor_, k=cfg.num_intrinsic_outliers
        )
    elif cfg.intrinsic_score == "dbscan":
        intrinsic_inds = dbscan_outlier_search(embeddings, cfg.num_intrinsic_outliers)
        # DBSCAN does not have an outlier score
        intrinsic_scores = np.zeros(len(intrinsic_inds))  # type: ignore[arg-type]
    elif cfg.intrinsic_score == "dbscan_lof_outlier":
        try:
            intrinsic_inds = dbscan_outlier_search(
                embeddings, cfg.num_intrinsic_outliers
            )
        except ValueError:
            print("WARNING: Could not find outliers with DBSCAN")
            # Default to the "lof" intrinsic score if DBSCAN doesn't find outliers
            intrinsic_inds = np.arange(len(embeddings))  # type: ignore[arg-type]
        clf = LocalOutlierFactor()
        clf.fit_predict(embeddings[intrinsic_inds])  # type: ignore[index]
        intrinsic_scores: "npt.ArrayLike" = clf.negative_outlier_factor_  # type: ignore[no-redef]
        # Sort the DBSCAN outliers by LOF score.
        # The smaller the lof_score, the more likely the point is an outlier.
        sorted_lof_inds = np.argsort(intrinsic_scores)
        intrinsic_inds = intrinsic_inds[sorted_lof_inds]  # type: ignore[call-overload, index]

    else:
        # If no intrinsic_score, simply return all the data
        intrinsic_inds = np.arange(len(embeddings))  # type: ignore[arg-type]
        intrinsic_scores = np.zeros(len(embeddings))  # type: ignore[arg-type]

    # Returns n_outlier best outliers sorted from best to worst
    return intrinsic_scores, intrinsic_inds


def get_extrinsic_score(
    intrinsic_inds: "npt.ArrayLike", virtual_h5_file: Path, cfg: OutlierDetectionConfig
) -> Tuple["npt.ArrayLike", "npt.ArrayLike"]:

    if cfg.extrinsic_score == "rmsd":
        # Get all RMSD values from virutal HDF5 file
        rmsds = parse_h5(virtual_h5_file, fields=["rmsd"])["rmsd"]
        # Select the subset choosen with the intrinsic score method
        intrinsic_rmsds = rmsds[intrinsic_inds]  # type: ignore[index]
        # Find the best points within the selected subset
        extrinsic_scores, extrinsic_inds = bestk(
            intrinsic_rmsds, k=cfg.num_extrinsic_outliers
        )
    else:
        # If no extrinsic_score, simply return the intrinsic selection
        extrinsic_inds = np.arange(cfg.num_extrinsic_outliers)
        extrinsic_scores = np.zeros(len(extrinsic_inds))  # type: ignore[arg-type]

    return extrinsic_scores, extrinsic_inds


def generate_outliers(
    md_data: Dict[str, List[str]],
    sampled_h5_files: List[str],
    outlier_inds: "npt.ArrayLike",
    intrinsic_scores: "npt.ArrayLike",
    extrinsic_scores: "npt.ArrayLike",
) -> List[Dict[str, object]]:

    assert len(intrinsic_scores) == len(extrinsic_scores)  # type: ignore[arg-type]
    assert len(intrinsic_scores) == len(outlier_inds)  # type: ignore[arg-type]

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
        outlier_inds, intrinsic_scores, extrinsic_scores  # type: ignore[arg-type, misc]
    ):
        # divmod returns a tuple of quotient and remainder
        sampled_index, frame = divmod(outlier_ind, cfg.n_traj_frames)  # type: ignore[operator]
        # Need to remap subsampled h5_file index back to all md_data
        all_index = h5_sample_ind_to_all[sampled_h5_files[sampled_index]]

        # Collect data to be passed into DeepDriveMD_API.write_task_json()
        # Data must be JSON serializable.
        outlier = {
            "structure_file": str(all_pdb_files[all_index]),
            "traj_file": str(all_traj_files[all_index]),
            "frame": int(frame),
            "outlier_ind": int(outlier_ind),  # type: ignore[call-overload]
            "intrinsic_score": float(intrinsic_score),  # type: ignore[arg-type]
            "extrinsic_score": float(extrinsic_score),  # type: ignore[arg-type]
        }
        outliers.append(outlier)

    return outliers


def main(cfg: OutlierDetectionConfig, encoder_gpu: int, distributed: bool) -> None:

    comm = setup_mpi_comm(distributed)
    comm_size, comm_rank = setup_mpi(comm)

    if comm_rank == 0:

        # Collect training data
        api = DeepDriveMD_API(cfg.experiment_directory)

        with Timer("agent_get_last_n_md_runs"):
            md_data = api.get_last_n_md_runs()

        with Timer("agent_get_virtual_h5_file"):
            virtual_h5_file, sampled_h5_files = get_virtual_h5_file(
                output_path=cfg.output_path,
                all_h5_files=md_data["data_files"],
                last_n=cfg.n_most_recent_h5_files,
                k_random_old=cfg.k_random_old_h5_files,
                virtual_name=f"virtual_{api.agent_stage.unique_name(cfg.output_path)}",
                node_local_path=cfg.node_local_path,
            )

        with open(cfg.output_path.joinpath("virtual-h5-metadata.json"), "w") as f:
            json.dump(sampled_h5_files, f)

        # Get best model hyperparameters and weights
        with Timer("agent_get_model_path"):
            token = get_model_path(api=api)
            assert token is not None
            model_cfg_path, model_weights_path = token

    else:
        virtual_h5_file, model_cfg_path, model_weights_path = None, None, None  # type: ignore[assignment]

    if comm_size > 1:
        virtual_h5_file = comm.bcast(virtual_h5_file, 0)  # type: ignore[union-attr]
        model_cfg_path = comm.bcast(model_cfg_path, 0)  # type: ignore[union-attr]
        model_weights_path = comm.bcast(model_weights_path, 0)  # type: ignore[union-attr]

    # Select machine learning model and generate embeddings
    with Timer("agent_get_representation"):
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

        with Timer("agent_get_intrinsic_score"):
            intrinsic_scores, intrinsic_inds = get_intrinsic_score(embeddings, cfg)

        # Prune the best intrinsically ranked points with an extrinsic score
        with Timer("agent_get_extrinsic_score"):
            extrinsic_scores, extrinsic_inds = get_extrinsic_score(
                intrinsic_inds, virtual_h5_file, cfg
            )

        # Take the subset of indices selected by the extrinsic method
        pruned_intrinsic_scores = intrinsic_scores[extrinsic_inds]  # type: ignore[index]
        pruned_intrinsic_inds = intrinsic_inds[extrinsic_inds]  # type: ignore[index]

        with Timer("agent_generate_outliers"):
            outliers = generate_outliers(
                md_data,
                sampled_h5_files,
                pruned_intrinsic_inds,
                pruned_intrinsic_scores,
                extrinsic_scores,
            )

        # Dump metadata to disk for MD stage
        with Timer("agent_write_task_json"):
            api.agent_stage.write_task_json(outliers, cfg.stage_idx, cfg.task_idx)

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
    # torch.multiprocessing.set_start_method("forkserver", force=True)
    with Timer("agent_stage"):
        args = parse_args()
        cfg = OutlierDetectionConfig.from_yaml(args.config)
        main(cfg, args.encoder_gpu, args.distributed)
