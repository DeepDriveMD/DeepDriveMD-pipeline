import os
import re
import click
from os.path import join
import wandb

# torch stuff
# from torchsummary import summary
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

# molecules stuff
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import (
    LossCallback,
    CheckpointCallback,
    SaveEmbeddingsCallback,
    TSNEPlotCallback,
)
from molecules.ml.unsupervised.point_autoencoder import AAE3d, AAE3dHyperparams


# mpi4py
import mpi4py

mpi4py.rc.initialize = False
from mpi4py import MPI  # noqa: E402


def parse_dict(ctx, param, value):
    if value is not None:
        token = value.split(",")
        result = {}
        for item in token:
            k, v = item.split("=")
            result[k] = v
        return result


def get_dataset(
    dataset_location,
    input_path,
    dataset_name,
    rmsd_name,
    fnc_name,
    num_points,
    num_features,
    split,
    shard_id=0,
    num_shards=1,
    normalize="box",
    cms_transform=False,
):

    if dataset_location == "storage":
        # Load training and validation data
        from molecules.ml.datasets import PointCloudDataset

        dataset = PointCloudDataset(
            input_path,
            dataset_name,
            rmsd_name,
            fnc_name,
            num_points,
            num_features,
            split=split,
            normalize=normalize,
            cms_transform=cms_transform,
        )

        # split across nodes
        if num_shards > 1:
            chunksize = len(dataset) // num_shards
            dataset = Subset(
                dataset, list(range(chunksize * shard_id, chunksize * (shard_id + 1)))
            )

    elif dataset_location == "cpu-memory":
        from molecules.ml.datasets import PointCloudInMemoryDataset

        dataset = PointCloudInMemoryDataset(
            input_path,
            dataset_name,
            rmsd_name,
            fnc_name,
            num_points,
            num_features,
            split=split,
            shard_id=shard_id,
            num_shards=num_shards,
            normalize=normalize,
            cms_transform=cms_transform,
        )

    else:
        raise NotImplementedError(
            f"Error, dataset_location = {dataset_location} not implemented."
        )

    return dataset


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to file containing preprocessed contact matrix data",
)
@click.option(
    "-dn",
    "--dataset_name",
    required=True,
    type=str,
    help="Name of the dataset in the HDF5 file.",
)
@click.option(
    "-rn",
    "--rmsd_name",
    default="rmsd",
    type=str,
    help="Name of the RMSD data in the HDF5 file.",
)
@click.option(
    "-fn",
    "--fnc_name",
    default="fnc",
    type=str,
    help="Name of the RMSD data in the HDF5 file.",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    required=True,
    type=click.Path(exists=True),
    help="Output directory for model data",
)
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(exists=True),
    help="Model checkpoint file to resume training. "
    "Checkpoint files saved as .pt by CheckpointCallback.",
)
@click.option("-r", "--resume", is_flag=True, help="Resume from latest checkpoint")
@click.option(
    "-iw",
    "--init_weights",
    type=click.Path(exists=True),
    help="Model checkpoint file to load model weights fromg. "
    "Checkpoint files saved as .pt by CheckpointCallback.",
)
@click.option(
    "-m", "--model_id", required=True, type=str, help="Model ID in for file naming"
)
@click.option(
    "-np", "--num_points", required=True, type=int, help="number of input points"
)
@click.option(
    "-nf",
    "--num_features",
    default=1,
    type=int,
    help="number of features per point in addition to 3D coordinates",
)
@click.option(
    "-eks",
    "--encoder_kernel_sizes",
    default=[5, 5, 3, 1, 1],
    type=int,
    nargs=5,
    help="list of encoder kernel sizes",
)
@click.option("-E", "--encoder_gpu", default=None, type=int, help="Encoder GPU id")
@click.option("-G", "--generator_gpu", default=None, type=int, help="Generator GPU id")
@click.option(
    "-D", "--discriminator_gpu", default=None, type=int, help="Discriminator GPU id"
)
@click.option(
    "-e", "--epochs", default=10, type=int, help="Number of epochs to train for"
)
@click.option(
    "-b", "--batch_size", default=128, type=int, help="Batch size for training"
)
@click.option("-opt", "--optimizer", callback=parse_dict, help="Optimizer parameters")
@click.option(
    "-d",
    "--latent_dim",
    default=256,
    type=int,
    help="Number of dimensions in latent space",
)
@click.option("-lw", "--loss_weights", callback=parse_dict, help="Loss parameters")
@click.option(
    "-ei",
    "--embed_interval",
    default=1,
    type=int,
    help="Saves embeddings every interval'th point",
)
@click.option(
    "-ti",
    "--tsne_interval",
    default=1,
    type=int,
    help="Saves model checkpoints, embeddings, tsne plots every " "interval'th point",
)
@click.option(
    "-S",
    "--sample_interval",
    default=20,
    type=int,
    help="For embedding plots. Plots every sample_interval'th point",
)
@click.option(
    "-wp",
    "--wandb_project_name",
    default=None,
    type=str,
    help="Project name for wandb logging",
)
@click.option("--distributed", is_flag=True, help="Enable distributed training")
@click.option(
    "-ndw",
    "--num_data_workers",
    default=0,
    type=int,
    help="Number of data loaders for training",
)
@click.option(
    "-dl",
    "--dataset_location",
    default="storage",
    type=str,
    help="String which specifies from where to feed the dataset. Valid choices are `storage` and `cpu-memory`.",
)
def main(
    input_path,
    dataset_name,
    rmsd_name,
    fnc_name,
    out_path,
    checkpoint,
    resume,
    init_weights,
    model_id,
    num_points,
    num_features,
    encoder_kernel_sizes,
    encoder_gpu,
    generator_gpu,
    discriminator_gpu,
    epochs,
    batch_size,
    optimizer,
    latent_dim,
    loss_weights,
    embed_interval,
    tsne_interval,
    sample_interval,
    wandb_project_name,
    distributed,
    num_data_workers,
    dataset_location,
):

    # do some scaffolding for DDP
    comm_rank = 0
    comm_size = 1
    comm = None
    if distributed and dist.is_available():
        # init mpi4py:
        MPI.Init_thread()

        # get communicator: duplicate from comm world
        comm = MPI.COMM_WORLD.Dup()

        # now match ranks between the mpi comm and the nccl comm
        os.environ["WORLD_SIZE"] = str(comm.Get_size())
        os.environ["RANK"] = str(comm.Get_rank())

        # init pytorch
        dist.init_process_group(backend="nccl", init_method="env://")
        comm_rank = dist.get_rank()
        comm_size = dist.get_world_size()

    # HP
    # model
    aae_hparams = {
        "num_features": num_features,
        "latent_dim": latent_dim,
        "encoder_kernel_sizes": encoder_kernel_sizes,
        "noise_std": 0.2,
        "lambda_rec": float(loss_weights["lambda_rec"]),
        "lambda_gp": float(loss_weights["lambda_gp"]),
    }
    hparams = AAE3dHyperparams(**aae_hparams)

    # optimizers
    optimizer_hparams = OptimizerHyperparams(
        name=optimizer["name"], hparams={"lr": float(optimizer["lr"])}
    )

    # create a dir for storing the model
    model_path = join(out_path, f"model-{model_id}")

    # Save hparams to disk
    if comm_rank == 0:
        os.makedirs(model_path, exist_ok=True)
        hparams.save(join(model_path, "model-hparams.json"))
        optimizer_hparams.save(join(model_path, "optimizer-hparams.json"))

    # construct model
    aae = AAE3d(
        num_points,
        num_features,
        batch_size,
        hparams,
        optimizer_hparams,
        gpu=(encoder_gpu, generator_gpu, discriminator_gpu),
        init_weights=init_weights,
    )

    enc_device = torch.device(f"cuda:{encoder_gpu}")
    if comm_size > 1:
        if (encoder_gpu == generator_gpu) and (encoder_gpu == discriminator_gpu):
            aae.model = DDP(
                aae.model, device_ids=[enc_device], output_device=enc_device
            )
        else:
            aae.model = DDP(aae.model, device_ids=None, output_device=None)

    # set global default device
    torch.cuda.set_device(enc_device.index)

    if comm_rank == 0:
        # Diplay model
        print(aae)

        # Only print summary when encoder_gpu is None or 0
        # summary(aae.model, (3 + num_features, num_points))

    # set up dataloaders
    train_dataset = get_dataset(
        dataset_location,
        input_path,
        dataset_name,
        rmsd_name,
        fnc_name,
        num_points,
        num_features,
        split="train",
        shard_id=comm_rank,
        num_shards=comm_size,
        normalize="box",
        cms_transform=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_data_workers,
    )

    valid_dataset = get_dataset(
        dataset_location,
        input_path,
        dataset_name,
        rmsd_name,
        fnc_name,
        num_points,
        num_features,
        split="valid",
        shard_id=comm_rank,
        num_shards=comm_size,
        normalize="box",
        cms_transform=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_data_workers,
    )

    print(
        f"Having {len(train_dataset)} training and {len(valid_dataset)} validation samples."
    )

    # do we want wandb
    wandb_config = None
    if (comm_rank == 0) and (wandb_project_name is not None):
        wandb.init(
            project=wandb_project_name,
            name=model_id,
            id=model_id,
            dir=model_path,
            resume=False,
        )
        wandb_config = wandb.config

        # log HP
        wandb_config.num_points = num_points
        wandb_config.num_features = num_features
        wandb_config.latent_dim = latent_dim
        wandb_config.lambda_rec = hparams.lambda_rec
        wandb_config.lambda_gp = hparams.lambda_gp
        # noise
        wandb_config.noise_std = hparams.noise_std

        # optimizer
        wandb_config.optimizer_name = optimizer_hparams.name
        for param in optimizer_hparams.hparams:
            wandb_config["optimizer_" + param] = optimizer_hparams.hparams[param]

        # watch model
        wandb.watch(aae.model)

    # Optional callbacks
    loss_callback = LossCallback(
        join(model_path, "loss.json"), wandb_config=wandb_config, mpi_comm=comm
    )

    checkpoint_callback = CheckpointCallback(
        out_dir=join(model_path, "checkpoint"), mpi_comm=comm
    )

    save_callback = SaveEmbeddingsCallback(
        out_dir=join(model_path, "embeddings"),
        interval=embed_interval,
        sample_interval=sample_interval,
        mpi_comm=comm,
    )

    # TSNEPlotCallback requires SaveEmbeddingsCallback to run first
    tsne_callback = TSNEPlotCallback(
        out_dir=join(model_path, "embeddings"),
        projection_type="3d",
        target_perplexity=100,
        interval=tsne_interval,
        tsne_is_blocking=True,
        wandb_config=wandb_config,
        mpi_comm=comm,
    )

    # Train model with callbacks
    callbacks = [
        loss_callback,
        checkpoint_callback,
        save_callback,
        tsne_callback,
    ]

    # see if resume is set
    if resume and (checkpoint is None):
        clist = [
            x for x in os.listdir(join(model_path, "checkpoint")) if x.endswith(".pt")
        ]
        checkpoints = sorted(
            clist, key=lambda x: re.match(r"epoch-\d*?-(\d*?-\d*?).pt", x).groups()[0]
        )
        if checkpoints:
            checkpoint = join(model_path, "checkpoint", checkpoints[-1])
            if comm_rank == 0:
                print(f"Resuming from checkpoint {checkpoint}.")
        else:
            if comm_rank == 0:
                print(
                    f"No checkpoint files in directory {join(model_path, 'checkpoint')}, \
                       cannot resume training, will start from scratch."
                )

    # train model with callbacks
    aae.train(
        train_loader, valid_loader, epochs, checkpoint=checkpoint, callbacks=callbacks
    )

    # Save loss history to disk.
    if comm_rank == 0:
        loss_callback.save(join(model_path, "loss.json"))

        # Save final model weights to disk
        aae.save_weights(
            join(model_path, "encoder-weights.pt"),
            join(model_path, "generator-weights.pt"),
            join(model_path, "discriminator-weights.pt"),
        )

    # Output directory structure
    #  out_path
    # ├── model_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── decoder-weights.pt
    # │   ├── encoder-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.pkl
    # │   └── optimizer-hparams.pkl


if __name__ == "__main__":
    # set forkserver (needed for summit runs, may cause errors elsewhere)
    # torch.multiprocessing.set_start_method('forkserver', force = True)

    main()
