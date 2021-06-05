import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union
import numpy as np
import glob
import subprocess
import time
import sys
import os

from deepdrivemd.utils import Timer, timer
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.agents.stream.config import OutlierDetectionConfig
import tensorflow.keras.backend as K

import hashlib
import pickle
from OutlierDB import *
from lockfile import LockFile
from aggregator_reader import *
# from utils import predict_from_cvae, outliers_from_latent

import cupy as cp
from cuml import DBSCAN as DBSCAN

from deepdrivemd.models.keras_cvae_stream.model import conv_variational_autoencoder

from dask.distributed import Client, wait

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import MDAnalysis as mda

def build_model(cfg, model_path):
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
    cvae.load(model_path)
    return cvae

def wait_for_model(cfg):
    while(True):
        if(os.path.exists(cfg.best_model)):
            break
        print(f"No model {cfg.best_model}, sleeping"); sys.stdout.flush()
        time.sleep(cfg.timeout2)
    return cfg.best_model

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


def dirs(cfg):
    top_dir = cfg.output_path
    tmp_dir = f"{top_dir}/tmp"
    published_dir = f"{top_dir}/published_outliers"

    if(not os.path.exists(tmp_dir)):
         os.mkdir(tmp_dir)
    if(not os.path.exists(published_dir)):
         os.mkdir(published_dir)
    return top_dir, tmp_dir, published_dir

def predict(cfg, model_path, cvae_input):
    cvae = build_model(cfg, model_path)
    cm_predict = cvae.return_embeddings(cvae_input[0])
    del cvae 
    K.clear_session()
    return cm_predict

def outliers_from_latent(cm_predict, eps=0.35):
    cm_predict = cp.asarray(cm_predict)
    db = DBSCAN(eps=eps, min_samples=10, max_mbytes_per_batch=500).fit(cm_predict)
    db_label = db.labels_.to_array()
    outlier_list = np.where(db_label == -1)
    return outlier_list

def cluster(cfg, cm_predict, outlier_list, eps):
    outlier_count = cfg.outlier_count

    while outlier_count > 0:
        n_outlier = 0
        try:
            outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps)) 
            n_outlier = len(outliers)
        except Exception as e:
            print(e)
            print("No outliers found")

        print(f'eps = {eps}, number of outlier found: {n_outlier}')

        if n_outlier > cfg.outlier_max: 
            eps = eps + 0.09*random.random()
        elif n_outlier < cfg.outlier_min:
            eps = max(0.01, eps - 0.09*random.random())
        else: 
            outlier_list.append(outliers) 
            break 
        outlier_count -= 1
    return eps


def write_pdb_frame(frame, original_pdb, output_pdb_fn):
    pdb = PDBFile(original_pdb)
    f = open(output_pdb_fn, 'w')
    PDBFile.writeFile(pdb.getTopology(), frame, f)
    f.close()

def write_pdb(myframe, hash, myframe_v, pdb_file, outliers_pdb_path):
    outlier_pdb_file = f'{outliers_pdb_path}/{hash}.pdb'
    outlier_v_file = f'{outliers_pdb_path}/{hash}.npy'
    write_pdb_frame(myframe, pdb_file, outlier_pdb_file)
    np.save(outlier_v_file, myframe_v)
    return 0


def write_outliers(cfg, outlier_list, client, tmp_dir, cvae_input):
    outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 
    outliers_pdb_path = tmp_dir

    new_outliers_list = [] 

    futures = []
    for outlier in outlier_list_uni:
        futures.append(client.submit(write_pdb, cvae_input[1][outlier], 
                                     cvae_input[2][outlier], 
                                     cvae_input[3][outlier], cfg.init_pdb_file, outliers_pdb_path))
    wait(futures)

    while(len(futures) > 0):
        del futures[0]

    for outlier in outlier_list_uni:
        # myframe = cvae_input[1][outlier]
        # myframe_v = cvae_input[3][outlier]
        hash = cvae_input[2][outlier]
        outlier_pdb_file = f'{outliers_pdb_path}/{hash}.pdb'
        # outlier_v_file = f'{outliers_pdb_path}/{hash}.npy'
        new_outliers_list.append(outlier_pdb_file) 

    return new_outliers_list

def compute_rmsd(ref_pdb_file, restart_pdbs):
    print("ref_pdf_file = ", ref_pdb_file)
    print("restart_pdbs[0] = ", restart_pdbs[0])
    print("len(restart_pdbs) = ", len(restart_pdbs))
    while(True):
        try:
            outlier_traj = mda.Universe(restart_pdbs[0], restart_pdbs) 
            break
        except Exception as e:
            print("Crashing while computing RMSD")
            print(e)
            time.sleep(3)
    ref_traj = mda.Universe(ref_pdb_file) 
    R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
    R.run()    
    restart_pdbs1 = [(rmsd, pdb) for rmsd, pdb in sorted(zip(R.rmsd[:,2], restart_pdbs))] 
    return restart_pdbs1

def write_db(cfg, restart_pdb, restart_pdbs1, tmp_dir):
    outlier_db_fn = f'{tmp_dir}/OutlierDB.pickle'
    db = OutlierDB(tmp_dir, restart_pdbs1)
    with open(outlier_db_fn, 'wb') as f:
        pickle.dump(db, f)    
    return db

def publish(tmp_dir, published_dir):
    dbfn = f"{published_dir}/OutlierDB.pickle"
    subprocess.getstatusoutput(f"touch {dbfn}")

    mylock = LockFile(dbfn)

    mylock.acquire()
    print(subprocess.getstatusoutput(f"rm -rf {published_dir}/*"))
    print(subprocess.getstatusoutput("mv {tmp_dir}/* {published_dir}/"))
    mylock.release()

    return


def main(cfg: OutlierDetectionConfig):
    print(cfg)
    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)
    with Timer("wait_for_model"):
        model_path = wait_for_model(cfg)

    mystreams = STREAMS(adios_files_list, lastN = cfg.lastN, config = cfg.adios_xml, stream_name = "AggregatorOutput", batch = cfg.batch)

    client = Client(processes=True, n_workers=cfg.n_workers, local_directory='/tmp')

    top_dir, tmp_dir, published_dir = dirs(cfg)
    eps = cfg.init_eps

    j = 0

    while(True):
        print(f"outlier iteration {j}")

        timer("outlier_search_iteration", 1)
        
        with Timer("outlier_read"):
            cvae_input = mystreams.next()

        with Timer("outlier_predict"):
            cm_predict = predict(cfg, model_path, cvae_input)

        outlier_list = []
        with Timer("outlier_cluster"):
            eps = cluster(cfg, cm_predict, outlier_list, eps)

        with Timer("outlier_write"):
            restart_pdbs = write_outliers(outlier_list, client)

        if(len(restart_pdbs) == 0):
            print("No outliers found")
            j += 1
            continue

        with Timer("outlier_rmsd"):
            restart_pdbs1 = compute_rmsd(cfg.ref_pdb_file, restart_pdbs)

        with Timer("outlier_db"):
            db = write_db(restart_pdbs, restart_pdbs1)
        
        with Timer("outlier_publish"):
            publish(tmp_dir, published_dir)

        timer("outlier_search_iteration", -1)
        j += 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = OutlierDetectionConfig.from_yaml(args.config)
    main(cfg)