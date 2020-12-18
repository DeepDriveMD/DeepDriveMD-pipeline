import argparse
import itertools
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import concatenate_virtual_h5
from deepdrivemd.models.aae.config import AAEModelConfig
from deepdrivemd.outlier_detection.lof.config import LOFConfig
from molecules.ml.datasets import PointCloudDataset
from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
from molecules.ml.unsupervised.point_autoencoder.aae import Encoder
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader, Subset

PathLike = Union[str, Path]

# Helper function for LocalOutlierFactor
def topk(a, k):
    """
    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)
    k : int
        specifies which element to partition upon
    Returns
    -------
    np.ndarray of length k containing indices of input array a
    coresponding to the k smallest values in a.
    """
    return np.argpartition(a, k)[:k]


def setup_mpi_comm(distributed: bool):
    if distributed:
        # get communicator: duplicate from comm world
        from mpi4py import MPI

        return MPI.COMM_WORLD.Dup()
    return None


def setup_mpi(comm):
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    return comm_size, comm_rank


def shard_dataset(
    dataset: torch.utils.data.Dataset, comm_size: int, comm_rank: int
) -> torch.utils.data.Dataset:

    fullsize = len(dataset)
    chunksize = fullsize // comm_size
    start = comm_rank * chunksize
    end = start + chunksize
    subset_indices = list(range(start, end))
    # deal with remainder
    for idx, i in enumerate(range(comm_size * chunksize, fullsize)):
        if idx == comm_rank:
            subset_indices.append(i)
    # split the set
    dataset = Subset(dataset, subset_indices)

    return dataset


def get_virtual_h5_file(
    all_h5_files: List[str],
    last_n: int,
    k_random_old: int,
    output_path: Path,
    node_local_path: Optional[Path],
) -> Tuple[Path, List[str]]:

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
    virtual_h5_file = output_path.joinpath("virtual.h5")
    concatenate_virtual_h5(h5_files, virtual_h5_file.as_posix())

    # If node local storage optimization is available, then
    # copy all HDF5 files to node local storage and make a
    # separate virtual HDF5 file on node local storage.
    if node_local_path is not None:
        h5_files = [shutil.copy(f, node_local_path) for f in h5_files]
        virtual_h5_file = node_local_path.joinpath("virtual.h5")
        concatenate_virtual_h5(h5_files, virtual_h5_file.as_posix())

    # Returns node local virtual file if available
    return virtual_h5_file, h5_files


def generate_embeddings(
    cfg: LOFConfig,
    h5_file: PathLike,
    comm=None,
) -> np.ndarray:

    comm_size, comm_rank = setup_mpi(comm)

    if comm_rank == 0:
        t_start = time.time()  # Start timer
        print("Generating embeddings")

    model_cfg = AAEModelConfig.from_yaml(cfg.model_path)

    model_hparams = AAE3dHyperparams(
        num_features=model_cfg.num_features,
        encoder_filters=model_cfg.encoder_filters,
        encoder_kernel_sizes=model_cfg.encoder_kernel_sizes,
        generator_filters=model_cfg.generator_filters,
        discriminator_filters=model_cfg.discriminator_filters,
        latent_dim=model_cfg.latent_dim,
        encoder_relu_slope=model_cfg.encoder_relu_slope,
        generator_relu_slope=model_cfg.generator_relu_slope,
        discriminator_relu_slope=model_cfg.discriminator_relu_slope,
        use_encoder_bias=model_cfg.use_encoder_bias,
        use_generator_bias=model_cfg.use_generator_bias,
        use_discriminator_bias=model_cfg.use_discriminator_bias,
        noise_mu=model_cfg.noise_mu,
        noise_std=model_cfg.noise_std,
        lambda_rec=model_cfg.lambda_rec,
        lambda_gp=model_cfg.lambda_gp,
    )

    encoder = Encoder(
        cfg.num_points,
        model_cfg.num_features,
        model_hparams,
        cfg.weights_path.as_posix(),
    )

    dataset = PointCloudDataset(
        str(h5_file),
        "point_cloud",
        "rmsd",
        "fnc",
        cfg.num_points,
        model_hparams.num_features,
        split="train",
        split_ptc=1.0,
        normalize="box",
        cms_transform=False,
    )

    # shard the dataset
    if comm_size > 1:
        dataset = shard_dataset(dataset, comm_size, comm_rank)

    # Put encoder on specified CPU/GPU
    encoder.to(cfg.device)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.inference_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # Collect embeddings (requires shuffle=False)
    embeddings = []
    for i, (data, *_) in enumerate(data_loader):
        data = data.to(cfg.device)
        embeddings.append(encoder.encode(data).cpu().numpy())
        if (comm_rank == 0) and (i % 100 == 0):
            print(f"Batch {i}/{len(data_loader)}")

    if comm_size > 1:
        # gather results
        embeddings = comm.allgather(embeddings)
        embeddings = list(itertools.chain.from_iterable(embeddings))

    embeddings = np.concatenate(embeddings)

    if comm_rank == 0:
        print(f"Generating Embeddings Time: {time.time() - t_start}s")

    return embeddings


def local_outlier_factor(
    embeddings: np.ndarray, n_outliers: int = 500, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:

    t_start = time.time()  # Start timer
    print("Running LOF")

    # compute LOF
    clf = LocalOutlierFactor(**kwargs)
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    # Array with 1 if inlier, -1 if outlier
    clf.fit_predict(embeddings)

    # Only sorts 1 element of negative_outlier_factors_, namely the element
    # that is position k in the sorted array. The elements above and below
    # the kth position are partitioned but not sorted. Returns the indices
    # of the elements of left hand side of the parition i.e. the top k.
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
    all_h5_files = md_data["h5_files"]
    all_traj_files = md_data["traj_files"]
    all_pdb_files = md_data["pdb_files"]

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

        # Collect data to be passed into DeepDriveMD_API.write_pdb()
        outlier = {
            "pdb_file": all_pdb_files[all_index],
            "dcd_file": all_traj_files[all_index],
            "frame": frame,
            "outlier_ind": outlier_ind,
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
            md_data["h5_files"],
            last_n=cfg.last_n_h5_files,
            k_random_old=cfg.k_random_old_h5_files,
            output_path=cfg.output_path,
            node_local_path=cfg.node_local_path,
        )

    else:
        virtual_h5_file = None

    if comm_size > 1:
        virtual_h5_file = comm.bcast(virtual_h5_file, 0)

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings = generate_embeddings(cfg, virtual_h5_file, comm=comm)

    if comm_rank == 0:
        # Perform LocalOutlierFactor outlier detection on embeddings
        outlier_inds, _ = local_outlier_factor(
            embeddings,
            n_outliers=cfg.num_outliers,
            n_jobs=cfg.sklearn_num_jobs,
            comm=comm,
        )

        outliers = generate_outliers(md_data, sampled_h5_files, list(outlier_inds))

        # Dump metadata to disk for MD stage
        api.write_agent_json(outliers)

        print(f"Outlier Detection Time: {time.time() - t_start}s")

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
