import os

# import json
import argparse
import itertools
import time
import torch
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

# from deepdrivemd.models.aae.config import AAEModelConfig

# plotting
import matplotlib.pyplot as plt

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


def generate_embeddings(
    cfg: LOFConfig,
    comm=None,
):

    # start timer
    t_start = time.time()

    # communicator stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    model_cfg = AAEModelConfig.from_yaml(args.config)

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
        print("Generating Embeddings Time: {}s".format(t_end - t_start))

    return embeddings, indices


# LOF
def local_outlier_factor(
    embeddings, n_outliers=500, plot_dir=None, comm=None, **kwargs
):
    # start timer
    t_start = time.time()

    # mpi stuff
    # comm_size = 1
    comm_rank = 0
    if comm is not None:
        # comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # LOF
    if comm_rank == 0:
        print("Running LOF")

    # compute LOF
    clf = LocalOutlierFactor(**kwargs)
    # Array with 1 if inlier, -1 if outlier
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    clf.fit_predict(embeddings)

    # print the results
    if (plot_dir is not None) and (comm_rank == 0):
        # create directory
        os.makedirs(plot_dir, exist_ok=True)
        # plot
        _, ax = plt.subplots(1, 1, tight_layout=True)
        ax.hist(clf.negative_outlier_factor_, bins="fd")
        plt.savefig(os.path.join(plot_dir, "score_distribution.png"))
        plt.close()

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


def find_frame(traj_dict, frame_number=0):
    local_frame = frame_number
    for key in sorted(traj_dict):
        if local_frame - traj_dict[key] < 0:
            return local_frame, key
        else:
            local_frame -= traj_dict[key]
    raise Exception(
        "frame %d should not exceed the total number of frames, %d"
        % (frame_number, sum(np.array(traj_dict.values()).astype(int)))
    )


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

    # mpi stuff
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

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

    comm_size = 1
    comm_rank = 0
    comm = None
    if distributed:
        # get communicator: duplicate from comm world
        from mpi4py import MPI

        comm = MPI.COMM_WORLD.Dup()
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings, indices = generate_embeddings(cfg, comm=comm)

    # Perform LocalOutlierFactor outlier detection on embeddings
    if comm_rank == 0:
        outlier_inds, scores = local_outlier_factor(
            embeddings,
            n_outliers=cfg.num_outliers,
            plot_dir=os.path.join(cfg.output_path, "figures"),
            n_jobs=-1,
            comm=comm,
        )
    else:
        outlier_inds = None
        scores = None

    if comm_size > 1:
        outlier_inds = comm.bcast(outlier_inds, 0)
        scores = comm.bcast(scores, 0)

    # if comm_rank == 0:
    #     print("outlier_inds shape: ", outlier_inds.shape)
    #     for ind, score in zip(outlier_inds, scores):
    #         print(f"ind, score: {ind}, {score}")

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

    # end
    t_end = time.time()
    if comm_rank == 0:
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
