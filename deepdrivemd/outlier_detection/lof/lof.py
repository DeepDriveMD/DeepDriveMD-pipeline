import os

# import json
import argparse
import itertools
from pathlib import Path
from typing import Optional, Tuple
import time
import shutil
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
import MDAnalysis as mda
from torch.utils.data import DataLoader, Subset
from sklearn.neighbors import LocalOutlierFactor
from molecules.utils import open_h5
from molecules.ml.datasets import PointCloudDataset
from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
from molecules.ml.unsupervised.point_autoencoder.aae import Encoder

from deepdrivemd.outlier_detection.lof.config import LOFConfig
from deepdrivemd.models.aae.config import AAEModelConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import concatenate_virtual_h5


# from deepdrivemd.models.aae.config import AAEModelConfig


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


def get_virtual_h5_file(
    experiment_directory: Path,
    last_n: int,
    k_random_old: int,
    output_path: Path,
    node_local_path: Optional[Path],
) -> Path:

    # Collect training data
    api = DeepDriveMD_API(experiment_directory)
    md_data = api.get_last_n_md_runs()
    all_h5_files = md_data["h5_files"]

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
    virtual_h5_path = output_path.joinpath("virtual.h5")
    concatenate_virtual_h5(h5_files, virtual_h5_path.as_posix())

    # If node local storage optimization is available, then
    # copy all HDF5 files to node local storage and make a
    # separate virtual HDF5 file on node local storage.
    if node_local_path is not None:
        h5_files = [shutil.copy(f, node_local_path) for f in h5_files]
        virtual_h5_path = node_local_path.joinpath("virtual.h5")
        concatenate_virtual_h5(h5_files, virtual_h5_path.as_posix())

    # Returns node local virtual file if available
    return virtual_h5_path


def generate_embeddings(
    cfg: LOFConfig,
    comm=None,
) -> Tuple[np.ndarray, np.ndarray]:

    # start timer
    t_start = time.time()

    comm_size, comm_rank = setup_mpi(comm)

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
        cfg.num_points, model_cfg.num_features, model_hparams, cfg.weights_path
    )

    dataset = PointCloudDataset(
        cfg.input_path,
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

    # Collect embeddings and associated index into simulation trajectory
    if comm_rank == 0:
        print("Generating embeddings")
    embeddings, indices = [], []
    for i, (data, rmsd, fnc, index) in enumerate(data_loader):
        data = data.to(cfg.device)
        embeddings.append(encoder.encode(data).cpu().numpy())
        indices.append(index)
        if (i % 100 == 0) and (comm_rank == 0):
            print(f"Batch {i}/{len(data_loader)}")

    if comm_size > 1:
        # gather results
        embeddings = comm.allgather(embeddings)
        embeddings = list(itertools.chain.from_iterable(embeddings))
        indices = comm.allgather(indices)
        indices = list(itertools.chain.from_iterable(indices))

    # concatenate
    embeddings = np.concatenate(embeddings)
    indices = np.concatenate(indices)

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print(f"Generating Embeddings Time: {t_end - t_start}s")

    return embeddings, indices


# LOF
def local_outlier_factor(embeddings, n_outliers=500, comm=None, **kwargs):
    # start timer
    t_start = time.time()

    # mpi stuff
    comm_size, comm_rank = setup_mpi(comm)

    # LOF
    if comm_rank == 0:
        print("Running LOF")

    # compute LOF
    clf = LocalOutlierFactor(**kwargs)
    # Array with 1 if inlier, -1 if outlier
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    clf.fit_predict(embeddings)

    # Only sorts 1 element of negative_outlier_factors_, namely the element
    # that is position k in the sorted array. The elements above and below
    # the kth position are partitioned but not sorted. Returns the indices
    # of the elements of left hand side of the parition i.e. the top k.
    outlier_inds = topk(clf.negative_outlier_factor_, k=n_outliers)

    outlier_scores = clf.negative_outlier_factor_[outlier_inds]

    # Only sorts an array of size n_outliers
    sort_inds = np.argsort(outlier_scores)

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("LOF Time: {}s".format(t_end - t_start))

    # Returns n_outlier best outliers sorted from best to worst
    return outlier_inds[sort_inds], outlier_scores[sort_inds]


def find_frame(index, traj_files, sim_lens):
    remainder = index
    for traj_file, sim_len in zip(traj_files, sim_lens):
        if remainder < sim_len:
            return traj_file, remainder
        remainder -= sim_len
    raise ValueError("Did not find frame")


def find_values(rewards_df, fname, xmin, xmax):
    selection = rewards_df["rewarded_inds"].between(xmin, xmax)
    # sel = rewarded_inds_ts[rewarded_inds_ts.between(xmin, xmax)].dropna().values
    selection = rewards_df[selection].dropna()
    selection["rewarded_inds"] -= xmin
    selection["files"] = fname
    return selection


def write_rewarded_pdbs(
    rewarded_inds, scores, pdb_out_path, data_path, max_retry_count=5, comm=None
):
    # start timer
    t_start = time.time()

    comm_size, comm_rank = setup_mpi(comm)

    # Get list of simulation trajectory files (Assume all are equal length (ns))
    with open_h5(data_path) as h5_file:
        traj_files = np.array(h5_file["traj_file"])
        sim_lens = np.array(h5_file["sim_len"])

    # store file and indices in dataframe
    trajdf = pd.DataFrame({"files": traj_files, "lengths": sim_lens})

    # get a global start index
    trajdf["start_index"] = (
        trajdf.shift(1).fillna(0)["lengths"].cumsum(axis=0).astype(int)
    )
    rewards_df = (
        pd.DataFrame({"rewarded_inds": rewarded_inds, "scores": scores})
        .sort_values("rewarded_inds")
        .reset_index(drop=True)
    )
    trajdf["selected"] = trajdf.apply(
        lambda x: find_values(
            rewards_df,
            x["files"],
            x["start_index"],
            x["start_index"] + x["lengths"] - 1,
        ),
        axis=1,
    )
    trajdf = trajdf[trajdf["selected"].apply(lambda x: not x.empty)].reset_index()
    trajdf = (
        pd.concat(trajdf["selected"].tolist())
        .sort_values("scores", ascending=True)
        .reset_index(drop=True)
    )
    trajdf["order"] = trajdf.index

    # now, chunk the frame for each node
    if comm_size > 1:
        fullsize = trajdf.shape[0]
        chunksize = fullsize // comm_size
        start = chunksize * comm_rank
        end = start + chunksize
        remainderdf = trajdf.iloc[chunksize * comm_size :, :]
        trajdf = trajdf.iloc[start:end, :]
        for idx, (ind, row) in enumerate(remainderdf.iterrows()):
            if idx == comm_rank:
                trajdf = trajdf.append(row)

    # For documentation on mda.Writer methods see:
    #   https://www.mdanalysis.org/mdanalysis/documentation_pages/coordinates/PDB.html
    #   https://www.mdanalysis.org/mdanalysis/_modules/MDAnalysis/coordinates/PDB.html#PDBWriter._update_frame

    # now group by files to improve IO
    trajdf.sort_values(by=["files", "rewarded_inds"], inplace=True)

    # do the IO
    groups = trajdf.groupby("files")
    outlier_pdbs = []
    orders = []
    for traj_file, item in groups:

        # extract sim tag
        sim_id = os.path.splitext(os.path.basename(traj_file))[0]

        # we might need more than one retry
        retry_count = 0
        while retry_count < max_retry_count:
            try:
                sim_pdb = os.path.realpath(
                    glob(os.path.join(os.path.dirname(traj_file), "*.pdb"))[0]
                )
                load_trajec = time.time()
                u = mda.Universe(sim_pdb, traj_file)
                load_trajec = time.time() - load_trajec
                # print("Load trajectory time: {}s".format(load_trajec))
                break
            except IOError:
                retry_count += 1

        if retry_count < max_retry_count:
            save_trajec = time.time()
            orders += list(item["order"])
            for frame in item["rewarded_inds"]:
                out_pdb = os.path.abspath(
                    join(pdb_out_path, f"{sim_id}_{frame:06}.pdb")
                )
                with mda.Writer(out_pdb) as writer:
                    # Write a single coordinate set to a PDB file
                    writer._update_frame(u)
                    writer._write_timestep(u.trajectory[frame])
                outlier_pdbs.append(out_pdb)
            save_trajec = time.time() - save_trajec

    if comm_size > 1:
        outlier_pdbs = comm.allgather(outlier_pdbs)
        outlier_pdbs = list(itertools.chain.from_iterable(outlier_pdbs))
        orders = comm.allgather(orders)
        orders = list(itertools.chain.from_iterable(orders))

    # sort by order
    outlier_pdbs = [x[1] for x in sorted(zip(orders, outlier_pdbs))]

    # stop timer
    t_end = time.time()
    if comm_rank == 0:
        print("Write PDB Time: {}s".format(t_end - t_start))

    return outlier_pdbs


def main(cfg: LOFConfig, distributed: bool):

    # start timer
    t_start = time.time()

    comm = setup_mpi_comm(distributed)
    comm_size, comm_rank = setup_mpi(comm)

    if comm_rank == 0:
        h5_data_path = get_virtual_h5_file(
            cfg.experiment_directory,
            last_n=cfg.last_n_h5_files,
            k_random_old=cfg.k_random_old_h5_files,
            output_path=cfg.output_path,
            node_local_path=cfg.node_local_path,
        )
    else:
        h5_data_path = None

    if comm_size > 1:
        h5_data_path = comm.bcast(h5_data_path, 0)

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings, indices = generate_embeddings(cfg, comm=comm)

    # Perform LocalOutlierFactor outlier detection on embeddings
    if comm_rank == 0:
        outlier_inds, scores = local_outlier_factor(
            embeddings,
            n_outliers=cfg.num_outliers,
            n_jobs=-1,
            comm=comm,
        )
    else:
        outlier_inds, scores = None, None

    if comm_size > 1:
        outlier_inds = comm.bcast(outlier_inds, 0)
        scores = comm.bcast(scores, 0)

    # Map shuffled indices back to in-order MD frames
    simulation_inds = indices[outlier_inds]

    if comm_rank == 0:
        print("simulation_inds shape", simulation_inds.shape)

    # Write rewarded PDB files to shared path
    # outlier_pdbs = write_rewarded_pdbs(
    #     simulation_inds, scores, cfg.out_path, data_path, comm=comm
    # )

    # if comm_rank == 0:
    #     print("outlier_pdbs len: ", len(outlier_pdbs))
    #     print("restart_checkpnts len: ", len(restart_checkpnts))

    # if comm_rank == 0:
    #     restart_points = restart_checkpnts + outlier_pdbs

    # if comm_rank == 0:
    #     print("restart_points len: ", len(restart_points))
    #     print("restart_points: ", restart_points)

    # if comm_rank == 0:
    #     with open(restart_points_path, "w") as restart_file:
    #         json.dump(restart_points, restart_file)

    # final barrier
    comm.barrier()

    if comm_rank == 0:
        t_end = time.time()
        print("Outlier Detection Time: {}s".format(t_end - t_start))


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
