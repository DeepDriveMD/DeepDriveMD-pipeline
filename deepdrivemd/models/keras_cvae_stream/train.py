import time
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from deepdrivemd.utils import Timer, timer, cm_1Dto2D_format
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import get_virtual_h5_file
from deepdrivemd.selection.latest.select_model import get_model_path
from deepdrivemd.models.keras_cvae_stream.config import KerasCVAEModelConfig
from deepdrivemd.models.keras_cvae_stream.utils import sparse_to_dense
from deepdrivemd.models.keras_cvae_stream.model import conv_variational_autoencoder
import subprocess
import glob
from aggregator_reader import *
import os

def get_init_weights(cfg: KerasCVAEModelConfig) -> Optional[str]:
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


def wait_for_input(cfg):
    # Wait until the expected number of agg.bp exist
    while(True):
        bpfiles = glob.glob(cfg.agg_dir + "/*/*/agg.bp")
        if(len(bpfiles) == cfg.num_agg):
            break
        print("Waiting for {cfg.num_agg} agg.bp files")
        time.sleep(cfg.timeout1)

    print(f"bpfiles = {bpfiles}")

    # Wait for enough time steps in each bp file
    while(True):
        enough = True
        for bp in bpfiles:
            com = f"bpls {bp}"
            a = subprocess.getstatusoutput(com)
            if(a[0] != 0):
                enough = False
                print(f"Waiting, a = {a}, {bp}")
                break
            try:
                steps = int(a[1].split("\n")[0].split("*")[0].split(" ")[-1])
            except:
                steps = 0
                enough = False
            if(steps < cfg.min_step_increment):
                enough = False
                print(f"Waiting, steps = {steps}, {bp}")
                break
        if(enough):
            break
        else:
            time.sleep(cfg.timeout2)

    return bpfiles

def next_input(cfg, streams):
    with Timer("ml_read"):
        cm_data_input = streams.next_cm()
    with Timer("ml_format"):
        cm_data_input = cm_1Dto2D_format(cm_data_input)
    np.random.shuffle(cm_data_input)
    train_val_split = int(0.8 * len(cm_data_input))
    return cm_data_input[:train_val_split], cm_data_input[train_val_split:]

def build_model(cfg):
    with Timer("ml_conv_variational_autoencoder"):
        cvae = conv_variational_autoencoder(
            image_size=cfg.final_shape,
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

def main(cfg):
    cfg.checkpoint_dir = cfg.output_path/"checkpoints"
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    cfg.published_model_dir = cfg.output_path/"published_model"
    cfg.published_model_dir.mkdir(exist_ok=True)

    with Timer("ml_wait_for_input"):
        cfg.bpfiles = wait_for_input(cfg)

    streams = STREAMS(cfg.bpfiles, lastN=cfg.max_steps, config=cfg.adios_xml)

    cvae = build_model(cfg)

    i = 0
    while(True):
        print(f"ML iteration {i}")
        cm_data_train, cm_data_val = next_input(cfg, streams)

        with Timer("ml_train"):
            cvae.train(
                cm_data_train,
                validation_data=cm_data_val,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                checkpoint_path = cfg.checkpoint_dir,
            )

        subprocess.getstatusoutput(f"mv {cfg.checkpoint_dir}/best.h5 {cfg.published_model_dir}/")

        loss = cvae.history.val_losses[-1]
        if(loss < cfg.max_loss):
            cvae.load(f"{cfg.checkpoint_dir}/best.h5")
        else:
            cvae = build_model(cfg)

        i += 1
        print("="*30)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    with Timer("machine_learning_stage"):
        args = parse_args()
        cfg = KerasCVAEModelConfig.from_yaml(args.config)
        main(cfg)
