import glob
import itertools
import subprocess
import sys
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from mdlearn.data.utils import train_valid_split
from mdlearn.nn.models.aae.point_3d_aae import AAE3d
from mdlearn.utils import get_torch_optimizer, log_checkpoint
from torchsummary import summary
from tqdm import tqdm

from deepdrivemd.data.stream.aggregator_reader import Streams, StreamVariable
from deepdrivemd.data.stream.enumerations import DataStructure
from deepdrivemd.models.aae_stream.config import Point3dAAEConfig
from deepdrivemd.models.aae_stream.utils import PointCloudDatasetInMemory
from deepdrivemd.utils import Timer, parse_args, timer


def wait_for_input(cfg: Point3dAAEConfig) -> List[str]:
    """Wait for the expected number of sufficiently large agg.bp files to be produced.

    Returns
    -------
    List[str]
         List of paths to aggregated files.
    """

    # Wait for enough bpfiles
    while True:
        bpfiles = glob.glob(str(cfg.agg_dir / "*/*/agg_4ml.bp*"))
        print(bpfiles)
        sys.stdout.flush()
        if len(bpfiles) == cfg.num_agg:
            break
        print(f"Waiting for {cfg.num_agg} agg.bp files")
        time.sleep(cfg.timeout1)

    print(f"bpfiles = {bpfiles}")

    time.sleep(60 * 5)
    return bpfiles


def next_input(
    cfg: Point3dAAEConfig, streams: Streams
) -> Tuple[np.ndarray, np.ndarray]:
    """Read the next batch of contact maps from aggregated files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
          Training and validation sets.
    """
    while True:
        with Timer("ml_read"):

            z = streams.next()
            print("z=", z)
            print("type(z)=", type(z))
            pc_data_input = z["point_cloud"]

            print("type(pc_data_input) = ", type(pc_data_input))
            print("dir(pc_data_input) = ", dir(pc_data_input))
            print("pc_data_input.shape = ", pc_data_input.shape)
            print("pc_data_input.dtype = ", pc_data_input.dtype)

            if pc_data_input.shape[0] > 100:
                break
            print("Sleeping")
            time.sleep(60)

    sys.stdout.flush()
    pc_data_input = np.transpose(pc_data_input, [0, 2, 1])

    dataset = PointCloudDatasetInMemory(
        data=pc_data_input,
        cms_transform=cfg.cms_transform,
    )
    print(dataset[0]["X"].shape)
    print("Dataset size:", len(dataset))

    train_loader, valid_loader = train_valid_split(
        dataset,
        cfg.split_pct,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_data_workers,
        drop_last=True,
        pin_memory=True,
        #    persistent_workers=True,
        #    prefetch_factor=cfg.prefetch_factor,
    )

    print("len(train_loader) = ", len(train_loader))
    print("len(valid_loader) = ", len(valid_loader))
    print("cfg.split_pct = ", cfg.split_pct)

    return train_loader, valid_loader


def build_model(cfg: Point3dAAEConfig):
    device = torch.device("cuda:0")
    with Timer("ml_aae"):
        model = AAE3d(
            cfg.num_points,
            cfg.num_features,
            cfg.latent_dim,
            cfg.encoder_bias,
            cfg.encoder_relu_slope,
            cfg.encoder_filters,
            cfg.encoder_kernels,
            cfg.decoder_bias,
            cfg.decoder_relu_slope,
            cfg.decoder_affine_widths,
            cfg.discriminator_bias,
            cfg.discriminator_relu_slope,
            cfg.discriminator_affine_widths,
        )
        model = model.to(device)

        summary(model, (3 + cfg.num_features, cfg.num_points))

        disc_optimizer = get_torch_optimizer(
            cfg.disc_optimizer.name,
            cfg.disc_optimizer.hparams,
            model.discriminator.parameters(),
        )
        ae_optimizer = get_torch_optimizer(
            cfg.ae_optimizer.name,
            cfg.ae_optimizer.hparams,
            itertools.chain(model.encoder.parameters(), model.decoder.parameters()),
        )

    return model, disc_optimizer, ae_optimizer, device


def train(
    train_loader,
    model: AAE3d,
    disc_optimizer,
    ae_optimizer,
    device,
    cfg: Point3dAAEConfig,
):
    avg_disc_loss, avg_ae_loss = 0.0, 0.0
    # Create prior noise buffer array
    noise = torch.FloatTensor(cfg.batch_size, cfg.latent_dim).to(device)

    for batch in tqdm(train_loader):

        x = batch["X"].to(device, non_blocking=True)

        # Encoder/Discriminator forward
        # Get latent vectors
        z = model.encode(x)
        # Get prior noise
        noise.normal_(mean=cfg.noise_mu, std=cfg.noise_std)
        # Get discriminator logits
        real_logits = model.discriminate(noise)
        fake_logits = model.discriminate(z)
        # Discriminator loss
        critic_loss = model.critic_loss(real_logits, fake_logits)
        gp_loss = model.gp_loss(noise, z)
        disc_loss = critic_loss + cfg.lambda_gp * gp_loss

        # Discriminator backward
        disc_optimizer.zero_grad()
        model.discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_optimizer.step()

        # Decoder forward
        recon_x = model.decode(z)
        recon_loss = model.recon_loss(x, recon_x)

        # Discriminator forward
        fake_logit = model.discriminate(z)
        decoder_loss = model.decoder_loss(fake_logit)
        ae_loss = decoder_loss + cfg.lambda_rec * recon_loss

        # AE backward
        ae_optimizer.zero_grad()
        model.decoder.zero_grad()
        model.encoder.zero_grad()
        ae_loss.backward()

        # Collect loss
        avg_disc_loss += disc_loss.item()
        avg_ae_loss += ae_loss.item()

    avg_disc_loss /= len(train_loader)
    avg_ae_loss /= len(train_loader)

    return avg_disc_loss, avg_ae_loss


def validate(valid_loader, model: AAE3d, device, cfg: Point3dAAEConfig):
    scalars = defaultdict(list)
    latent_vectors = []
    avg_ae_loss = 0.0
    for batch in valid_loader:
        x = batch["X"].to(device)
        z = model.encode(x)
        recon_x = model.decode(z)
        avg_ae_loss += model.recon_loss(x, recon_x).item()

        # Collect latent vectors for visualization
        latent_vectors.append(z.cpu().numpy())
        for name in cfg.scalar_dset_names:
            scalars[name].append(batch[name].cpu().numpy())

    avg_ae_loss /= len(valid_loader)
    latent_vectors = np.concatenate(latent_vectors)
    scalars = {name: np.concatenate(scalar) for name, scalar in scalars.items()}

    return avg_ae_loss, latent_vectors, scalars


def train_model(
    model,
    ae_optimizer,
    disc_optimizer,
    train_loader,
    valid_loader,
    device,
    cfg: Point3dAAEConfig,
):
    best_valid_loss = np.inf
    for epoch in range(0, cfg.epochs):
        train_start = time.time()
        # Training
        model.train()
        avg_train_disc_loss, avg_train_ae_loss = train(
            train_loader, model, disc_optimizer, ae_optimizer, device, cfg
        )

        print(
            "====> Epoch: {} Train:\tAvg Disc loss: {:.4f}\tAvg AE loss: {:.4f}\tTime: {:.4f}".format(
                epoch, avg_train_disc_loss, avg_train_ae_loss, time.time() - train_start
            )
        )

        valid_start = time.time()
        # Validation
        model.eval()
        with torch.no_grad():
            avg_valid_recon_loss, _, _ = validate(valid_loader, model, device, cfg)

        print(
            "====> Epoch: {} Valid:\tAvg recon loss: {:.4f}\tTime: {:.4f}\n".format(
                epoch, avg_valid_recon_loss, time.time() - valid_start
            )
        )

        # Log checkpoint
        if avg_valid_recon_loss < best_valid_loss:
            best_valid_loss = avg_valid_recon_loss
            log_checkpoint(
                cfg.checkpoint_dir / "best.pt",
                epoch,
                model,
                {"disc_optimizer": disc_optimizer, "ae_optimizer": ae_optimizer},
            )
            print(
                f"Logging checkpoint at epoch {epoch} with validation loss {best_valid_loss}"
            )

        print("Total time: {:.4f}".format(time.time() - train_start))


def main(cfg: Point3dAAEConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.num_data_workers)

    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()

    cfg.checkpoint_dir = cfg.output_path / "checkpoints"
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    cfg.published_model_dir = cfg.output_path / "published_model"
    cfg.published_model_dir.mkdir(exist_ok=True)

    with Timer("ml_wait_for_input"):
        bpfiles = wait_for_input(cfg)

    print("In main, bpfiles = ", bpfiles)
    print(cfg.adios_xml_agg)
    print(cfg.max_steps)
    print(cfg.read_batch)
    sys.stdout.flush()

    bpfiles = list(map(lambda x: x.replace(".sst", ""), bpfiles))
    print("In main, after map, bpfiles = ", bpfiles)

    streams = Streams(
        bpfiles,
        [StreamVariable("point_cloud", np.float32, DataStructure.array)],
        lastN=cfg.max_steps,
        config=cfg.adios_xml_agg_4ml,
        batch=cfg.read_batch,
        stream_name="AggregatorOutput4ml",
        # change for different aggregators
    )

    model, disc_optimizer, ae_optimizer, device = build_model(cfg)

    # Infinite loop of AAE training
    # After training iteration, publish the model in the directory
    # from which it is picked up by outlier search
    for i in itertools.count(0):
        timer("ml_iteration", 1)
        print(f"ML iteration {i}")
        train_loader, valid_loader = next_input(cfg, streams)

        best_model_path = cfg.published_model_dir / "best.pt"
        if i > 0:
            checkpoint = torch.load(str(best_model_path), map_location="cpu")
            print("type(checkpoint)=", type(checkpoint))
            print("dir(checkpoint)=", dir(checkpoint))
            print("checkpoint.keys() = ", checkpoint.keys())
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            ae_optimizer.load_state_dict(checkpoint["ae_optimizer_state_dict"])
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
            model = model.to(device)
            print(f"Loaded checkpoint from previous iteration at epoch: {epoch}")

        with Timer("ml_train"):
            train_model(
                model,
                ae_optimizer,
                disc_optimizer,
                train_loader,
                valid_loader,
                device,
                cfg,
            )

        checkpoint_path = cfg.checkpoint_dir / "best.pt"

        if checkpoint_path.exists():
            # shutil.move(str(checkpoint_path), str(cfg.published_model_dir))
            subprocess.getstatusoutput(
                f"mv {checkpoint_path} {cfg.published_model_dir}/"
            )

        print("=" * 30)
        timer("ml_iteration", -1)


if __name__ == "__main__":
    args = parse_args()
    cfg = Point3dAAEConfig.from_yaml(args.config)

    print(cfg)
    main(cfg)
