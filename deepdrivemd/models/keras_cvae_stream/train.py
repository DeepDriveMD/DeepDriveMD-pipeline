import glob
import itertools
import os
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from deepdrivemd.data.stream.aggregator_reader import StreamContactMapVariable, Streams
from deepdrivemd.data.stream.enumerations import DataStructure
from deepdrivemd.models.keras_cvae.model import CVAE
from deepdrivemd.models.keras_cvae_stream.config import KerasCVAEModelConfig
from deepdrivemd.utils import Timer, parse_args, timer


def wait_for_input(cfg: KerasCVAEModelConfig) -> List[str]:
    """Wait for the expected number of sufficiently large agg.bp files to be produced.

    Returns
    -------
    List[str]
         List of paths to aggregated files.
    """

    # Wait for enough bpfiles
    while True:
        bpfiles = glob.glob(str(cfg.agg_dir / "*/*/agg.bp"))
        if len(bpfiles) == cfg.num_agg:
            break
        print(f"Waiting for {cfg.num_agg} agg.bp files")
        time.sleep(cfg.timeout1)

    print(f"bpfiles = {bpfiles}")

    # Wait for enough time steps in each bp file
    while True:
        enough = True
        for bp in bpfiles:
            com = f"bpls {bp}"
            a = subprocess.getstatusoutput(com)
            if a[0] != 0:
                enough = False
                print(f"Waiting, a = {a}, {bp}")
                break
            try:
                steps = int(a[1].split("\n")[0].split("*")[0].split(" ")[-1])
            except Exception as e:
                print("Exception ", e)
                steps = 0
                enough = False
            if steps < cfg.min_step_increment:
                enough = False
                print(f"Waiting, steps = {steps}, {bp}")
                break
        if enough:
            break
        else:
            time.sleep(cfg.timeout2)

    return bpfiles


def next_input(
    cfg: KerasCVAEModelConfig, streams: Streams
) -> Tuple[List[npt.ArrayLike], List[npt.ArrayLike]]:
    """Read the next batch of contact maps from aggregated files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
          Training and validation sets.
    """

    with Timer("ml_read"):
        cm_data_input = streams.next()[0]
    cm_data_input = np.expand_dims(cm_data_input, axis=-1)  # type: ignore

    cfg.initial_shape = cm_data_input.shape[1:3]  # type: ignore
    cfg.final_shape = list(cm_data_input.shape[1:3]) + [1]  # type: ignore

    print(
        f"in next_input: cm_data_input.shape = {cm_data_input.shape}"  # type: ignore
    )  # (2000, 28, 28, 1)
    np.random.shuffle(cm_data_input)  # type: ignore
    train_val_split = int(cfg.split_pct * len(cm_data_input))
    print(f"train_val_split = {train_val_split}")
    sys.stdout.flush()
    return cm_data_input[:train_val_split], cm_data_input[train_val_split:]


def build_model(cfg: KerasCVAEModelConfig) -> CVAE:
    with Timer("ml_conv_variational_autoencoder"):
        cvae = CVAE(
            image_size=cfg.final_shape,  # type: ignore
            channels=cfg.final_shape[-1],
            conv_layers=cfg.conv_layers,
            feature_maps=cfg.conv_filters,
            filter_shapes=cfg.conv_filter_shapes,
            strides=cfg.conv_strides,
            dense_layers=cfg.dense_layers,
            dense_neurons=cfg.dense_neurons,
            dense_dropouts=cfg.dense_dropouts,
            latent_dim=cfg.latent_dim,
        )
    cvae.model.summary()
    return cvae


def main(cfg: KerasCVAEModelConfig) -> None:
    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()

    cfg.checkpoint_dir = cfg.output_path / "checkpoints"
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    cfg.published_model_dir = cfg.output_path / "published_model"
    cfg.published_model_dir.mkdir(exist_ok=True)

    with Timer("ml_wait_for_input"):
        bpfiles = wait_for_input(cfg)

    streams = Streams(
        bpfiles,
        [StreamContactMapVariable("contact_map", np.uint8, DataStructure.array)],
        lastN=cfg.max_steps,
        config=cfg.adios_xml_agg,
        batch=cfg.read_batch,
    )

    # Infinite loop of CVAE training
    # After training iteration, publish the model in the directory from which it is picked up by outlier search
    for i in itertools.count(0):
        timer("ml_iteration", 1)
        print(f"ML iteration {i}")
        cm_data_train, cm_data_val = next_input(cfg, streams)

        if "cvae" not in locals():
            cvae = build_model(cfg)

        with Timer("ml_train"):
            cvae.train(
                cm_data_train,  # type: ignore
                validation_data=cm_data_val,  # type: ignore
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                checkpoint_path=cfg.checkpoint_dir,
            )

        loss = cvae.history.val_losses[-1]
        print("loss = ", loss)
        best_model = f"{cfg.checkpoint_dir}/best.h5"

        if cfg.reinit or loss > cfg.max_loss:
            del cvae
            cvae = build_model(cfg)
        else:
            cvae.load(best_model)

        if loss < cfg.max_loss and os.path.exists(best_model):
            subprocess.getstatusoutput(
                f"mv {cfg.checkpoint_dir}/best.h5 {cfg.published_model_dir}/"
            )

        print("=" * 30)
        timer("ml_iteration", -1)


if __name__ == "__main__":
    args = parse_args()
    cfg = KerasCVAEModelConfig.from_yaml(args.config)
    main(cfg)
