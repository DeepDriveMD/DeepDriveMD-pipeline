import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import wandb
from deepdrivemd.models.aae.config import AAEModelConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file
from deepdrivemd.selection.latest.select_model import get_model_path

# torch stuff
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


def setup_wandb(
    cfg: AAEModelConfig, model: torch.nn.Module, comm_rank: int
) -> Optional[wandb.config]:
    # Setup wandb
    wandb_config = None
    if (comm_rank == 0) and (cfg.wandb_project_name is not None):
        wandb.init(
            project=cfg.wandb_project_name,
            name=cfg.model_tag,
            id=cfg.model_tag,
            dir=cfg.output_path.as_posix(),
            config=cfg.dict(),
            resume=False,
        )
        wandb_config = wandb.config
        # watch model
        wandb.watch(model)

    return wandb_config


def get_h5_training_file(cfg: AAEModelConfig) -> Tuple[Path, List[str]]:
    # Collect training data
    api = DeepDriveMD_API(cfg.experiment_directory)
    md_data = api.get_last_n_md_runs()
    all_h5_files = md_data["data_files"]

    virtual_h5_path, h5_files = get_virtual_h5_file(
        output_path=cfg.output_path,
        all_h5_files=all_h5_files,
        last_n=cfg.last_n_h5_files,
        k_random_old=cfg.k_random_old_h5_files,
        virtual_name=f"virtual_{cfg.model_tag}",
        node_local_path=cfg.node_local_path,
    )

    return virtual_h5_path, h5_files


def get_init_weights(cfg: AAEModelConfig) -> Optional[str]:
    if cfg.init_weights_path is None:

        if cfg.stage_idx == 0:
            # Case for first iteration with no pretrained weights
            return

        token = get_model_path(
            stage_idx=cfg.stage_idx - 1, experiment_dir=cfg.experiment_directory
        )
        if token is None:
            # Case for no pretrained weights
            return
        else:
            # Case where model selection has run before
            _, init_weights = token
    else:
        # Case for pretrained weights
        init_weights = cfg.init_weights_path

    return init_weights.as_posix()


def get_dataset(
    dataset_location: str,
    input_path: Path,
    dataset_name: str,
    rmsd_name: str,
    fnc_name: str,
    num_points: int,
    num_features: int,
    split: str,
    shard_id: int = 0,
    num_shards: int = 1,
    normalize: str = "box",
    cms_transform: bool = False,
) -> torch.utils.data.Dataset:

    if dataset_location == "storage":
        # Load training and validation data
        from molecules.ml.datasets import PointCloudDataset

        dataset = PointCloudDataset(
            input_path.as_posix(),
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
            input_path.as_posix(),
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
        raise ValueError(f"Invalid option for dataset_location: {dataset_location}")

    return dataset


def main(
    cfg: AAEModelConfig,
    encoder_gpu: int,
    generator_gpu: int,
    discriminator_gpu: int,
    distributed: bool,
):

    # Do some scaffolding for DDP
    comm_rank = 0
    comm_size = 1
    comm = None
    if distributed and dist.is_available():

        import mpi4py

        mpi4py.rc.initialize = False
        from mpi4py import MPI  # noqa: E402

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

    model_hparams = AAE3dHyperparams(
        num_features=cfg.num_features,
        encoder_filters=cfg.encoder_filters,
        encoder_kernel_sizes=cfg.encoder_kernel_sizes,
        generator_filters=cfg.generator_filters,
        discriminator_filters=cfg.discriminator_filters,
        latent_dim=cfg.latent_dim,
        encoder_relu_slope=cfg.encoder_relu_slope,
        generator_relu_slope=cfg.generator_relu_slope,
        discriminator_relu_slope=cfg.discriminator_relu_slope,
        use_encoder_bias=cfg.use_encoder_bias,
        use_generator_bias=cfg.use_generator_bias,
        use_discriminator_bias=cfg.use_discriminator_bias,
        noise_mu=cfg.noise_mu,
        noise_std=cfg.noise_std,
        lambda_rec=cfg.lambda_rec,
        lambda_gp=cfg.lambda_gp,
    )

    # optimizers
    optimizer_hparams = OptimizerHyperparams(
        name=cfg.optimizer_name, hparams={"lr": cfg.optimizer_lr}
    )

    # Save hparams to disk and load initial weights and create virtual h5 file
    if comm_rank == 0:
        cfg.output_path.mkdir(exist_ok=True)
        model_hparams.save(cfg.output_path.joinpath("model-hparams.json"))
        optimizer_hparams.save(cfg.output_path.joinpath("optimizer-hparams.json"))
        init_weights = get_init_weights(cfg)
        h5_file, h5_files = get_h5_training_file(cfg)
        with open(cfg.output_path.joinpath("virtual-h5-metadata.json"), "w") as f:
            json.dump(h5_files, f)

    else:
        init_weights, h5_file = None, None

    if comm_size > 1:
        init_weights = comm.bcast(init_weights, 0)
        h5_file = comm.bcast(h5_file, 0)

    # construct model
    aae = AAE3d(
        cfg.num_points,
        cfg.num_features,
        cfg.batch_size,
        model_hparams,
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

    assert isinstance(h5_file, Path)
    # set up dataloaders
    train_dataset = get_dataset(
        cfg.dataset_location,
        h5_file,
        cfg.dataset_name,
        cfg.rmsd_name,
        cfg.fnc_name,
        cfg.num_points,
        cfg.num_features,
        split="train",
        shard_id=comm_rank,
        num_shards=comm_size,
        normalize="box",
        cms_transform=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_data_workers,
    )

    valid_dataset = get_dataset(
        cfg.dataset_location,
        h5_file,
        cfg.dataset_name,
        cfg.rmsd_name,
        cfg.fnc_name,
        cfg.num_points,
        cfg.num_features,
        split="valid",
        shard_id=comm_rank,
        num_shards=comm_size,
        normalize="box",
        cms_transform=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_data_workers,
    )

    print(
        f"Having {len(train_dataset)} training and {len(valid_dataset)} validation samples."
    )

    wandb_config = setup_wandb(cfg, aae.model, comm_rank)

    # Optional callbacks
    loss_callback = LossCallback(
        cfg.output_path.joinpath("loss.json"), wandb_config=wandb_config, mpi_comm=comm
    )

    checkpoint_callback = CheckpointCallback(
        out_dir=cfg.output_path.joinpath("checkpoint"), mpi_comm=comm
    )

    save_callback = SaveEmbeddingsCallback(
        out_dir=cfg.output_path.joinpath("embeddings"),
        interval=cfg.embed_interval,
        sample_interval=cfg.sample_interval,
        mpi_comm=comm,
    )

    # TSNEPlotCallback requires SaveEmbeddingsCallback to run first
    tsne_callback = TSNEPlotCallback(
        out_dir=cfg.output_path.joinpath("embeddings"),
        projection_type="3d",
        target_perplexity=100,
        interval=cfg.tsne_interval,
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

    # Optionaly train for a different number of
    # epochs on the first DDMD iterations
    if cfg.stage_idx == 0:
        epochs = cfg.initial_epochs
    else:
        epochs = cfg.epochs

    aae.train(train_loader, valid_loader, epochs, callbacks=callbacks)

    # Save loss history to disk.
    if comm_rank == 0:
        loss_callback.save(cfg.output_path.joinpath("loss.json"))

        # Save final model weights to disk
        aae.save_weights(
            cfg.output_path.joinpath("encoder-weights.pt"),
            cfg.output_path.joinpath("generator-weights.pt"),
            cfg.output_path.joinpath("discriminator-weights.pt"),
        )

    # Output directory structure
    #  out_path
    # ├── cfg.output_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── encoder-weights.pt
    # │   ├── generator-weights.pt
    # │   ├── discriminator-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.json
    # │   └── optimizer-hparams.json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    parser.add_argument(
        "-E", "--encoder_gpu", help="GPU to place encoder", type=int, default=0
    )
    parser.add_argument(
        "-G", "--generator_gpu", help="GPU to place generator", type=int, default=0
    )
    parser.add_argument(
        "-D", "--decoder_gpu", help="GPU to place decoder", type=int, default=0
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # set forkserver (needed for summit runs, may cause errors elsewhere)
    # torch.multiprocessing.set_start_method('forkserver', force = True)

    args = parse_args()
    cfg = AAEModelConfig.from_yaml(args.config)
    main(cfg, args.encoder_gpu, args.generator_gpu, args.decoder_gpu, args.distributed)
