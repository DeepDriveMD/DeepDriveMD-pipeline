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
import gc
from numba import cuda

from deepdrivemd.utils import Timer, timer, t1Dto2D
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.agents.stream.config import OutlierDetectionConfig
import tensorflow.keras.backend as K
from  MDAnalysis.analysis.rms import RMSD

import hashlib
import pickle
from OutlierDB import *
from lockfile import LockFile
from aggregator_reader import *

import cupy as cp
from cuml import DBSCAN as DBSCAN

# from cuml import TSNE
from sklearn.manifold import TSNE
import cuml

from deepdrivemd.models.keras_cvae_stream.model import conv_variational_autoencoder

from dask.distributed import Client, wait
from multiprocessing import Pool
import multiprocessing
from functools import partial



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
        print(f"Waiting for {cfg.num_agg} agg.bp files")
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

def predict(cfg, model_path, cvae_input, batch_size=32):
    cvae = build_model(cfg, model_path)
    input = np.expand_dims(cvae_input[0], axis = -1)

    cm_predict = cvae.return_embeddings(input, batch_size)
    del cvae 
    K.clear_session()
    return cm_predict

def outliers_from_latent(cm_predict, eps=0.35, min_samples=10):
    cm_predict = cp.asarray(cm_predict)
    db = DBSCAN(eps=eps, min_samples=min_samples, max_mbytes_per_batch=100).fit(cm_predict)
    db_label = db.labels_.to_array()
    print("unique labels = ", np.unique(db_label))
    outlier_list = np.where(db_label == -1)
    K.clear_session()
    return outlier_list

def cluster(cfg, cm_predict, outlier_list, eps, min_samples):
    outlier_count = cfg.outlier_count
    while outlier_count > 0:
        n_outlier = 0
        try:
            outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps, min_samples=min_samples)) 
            n_outlier = len(outliers)
        except Exception as e:
            print(e)
            print("No outliers found")

        print(f'eps = {eps}, min_samples = {min_samples}, number of outlier found: {n_outlier}')

        if n_outlier > cfg.outlier_max: 
            eps = eps + 0.09*random.random()
            min_samples -= int(random.random() < 0.5)
            min_samples = max(5, min_samples)
        elif n_outlier < cfg.outlier_min:
            eps = max(0.01, eps - 0.09*random.random())
            min_samples += int(random.random() < 0.5)
        else: 
            outlier_list.append(outliers) 
            break 
        outlier_count -= 1
    return eps, min_samples


def write_pdb_frame(frame, original_pdb, output_pdb_fn):
    pdb = PDBFile(original_pdb)
    f = open(output_pdb_fn, 'w')
    PDBFile.writeFile(pdb.getTopology(), frame, f)
    f.close()

def write_top_outliers(cfg, tmp_dir, top):
    positions = top[0]
    velocities = top[1]
    md5s = top[2]

    for p,v,m in zip(positions, velocities, md5s):
        outlier_pdb_file = f'{tmp_dir}/{m}.pdb'
        outlier_v_file = f'{tmp_dir}/{m}.npy'
        write_pdb_frame(p, cfg.init_pdb_file, outlier_pdb_file)
        np.save(outlier_v_file, v)

def write_db(top, tmp_dir):
    outlier_db_fn = f'{tmp_dir}/OutlierDB.pickle'
    outlier_files = list(map(lambda x: f'{tmp_dir}/{x}.pdb', top[2]))
    rmsds = top[3]
    db = OutlierDB(tmp_dir, list(zip(rmsds, outlier_files)))
    with open(outlier_db_fn, 'wb') as f:
        pickle.dump(db, f)    
    return db

def publish(tmp_dir, published_dir):
    dbfn = f"{published_dir}/OutlierDB.pickle"
    subprocess.getstatusoutput(f"touch {dbfn}")

    mylock = LockFile(dbfn)

    mylock.acquire()
    print(subprocess.getstatusoutput(f"rm -rf {published_dir}/*.pdb {published_dir}/*.npy"))
    print(subprocess.getstatusoutput(f"mv {tmp_dir}/* {published_dir}/"))
    mylock.release()

    return

def top_outliers(cfg, cvae_input, outlier_list):
    N = cfg.num_sim
    outlier_list = list(outlier_list[0])
    positions = cvae_input[1][outlier_list]
    velocities = cvae_input[3][outlier_list]
    md5s = cvae_input[2][outlier_list]
    rmsds = cvae_input[4][outlier_list]
    
    z = list(zip(positions, velocities, md5s, rmsds, outlier_list))
    z.sort(key = lambda x: x[3])
    z = z[:N]
    z = list(zip(*z))

    return z

def main(cfg: OutlierDetectionConfig):
    print(subprocess.getstatusoutput("hostname")[1]); sys.stdout.flush()

    print(cfg)
    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)
    with Timer("wait_for_model"):
        model_path = wait_for_model(cfg)

    mystreams = STREAMS(adios_files_list, lastN = cfg.lastN, config = cfg.adios_xml, stream_name = "AggregatorOutput", batch = cfg.batch)

    top_dir, tmp_dir, published_dir = dirs(cfg)
    eps = cfg.init_eps
    min_samples = cfg.init_min_samples

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
            eps, min_samples = cluster(cfg, cm_predict, outlier_list, eps, min_samples)

            if(len(outlier_list) == 0 or len(outlier_list[0]) < cfg.num_sim):
                j += 1
                print("No outliers found")
                time.sleep(30)
                timer("outlier_search_iteration", -1)
                continue

        top = top_outliers(cfg, cvae_input, outlier_list)
        print("top outliers = ", top[3])

        with Timer("outlier_write"):
            write_top_outliers(cfg, tmp_dir, top)

        with Timer("outlier_db"):
            db = write_db(top, tmp_dir)

        with Timer("outlier_publish"):
            publish(tmp_dir, published_dir)

        timer("outlier_search_iteration", -1)
        j += 1


def read_lastN(adios_files_list, lastN):
    variable_lists = {}
    for bp in adios_files_list:
        with adios2.open(bp,  "r") as fh:
            steps = fh.steps()
            start_step = steps - lastN - 2
            if(start_step < 0):
                start_step = 0
                lastN = steps
            for v in ['contact_map', 'rmsd']:
                if(v == 'contact_map'):
                    shape = list(map(int, fh.available_variables()[v]['Shape'].split(",")))
                elif(v == 'rmsd'):
                    print(fh.available_variables()[v]['Shape'])
                    sys.stdout.flush()
                if(v == 'contact_map'):
                    start = [0]*len(shape)
                    var = fh.read(v, start = start, count = shape, step_start = start_step, step_count = lastN)
                elif(v == 'rmsd'):
                    var = fh.read(v, [], [], step_start = start_step, step_count = lastN)
                print("v = ", v, " var.shape = ", var.shape)
                try:
                    variable_lists[v].append(var)
                except:
                    variable_lists[v] = [var]

    for vl in variable_lists:
        print(vl)
        print(len(variable_lists[vl]))
        variable_lists[vl] = np.vstack(variable_lists[vl])
        print(len(variable_lists[vl]))


    variable_lists['contact_map'] = np.array(list(map(lambda x: t1Dto2D(np.unpackbits(x.astype('uint8'))), list(variable_lists['contact_map']))))


    print(variable_lists['contact_map'].shape)
    print(variable_lists['rmsd'].shape)
    sys.stdout.flush()
    return variable_lists['contact_map'], np.concatenate(variable_lists['rmsd'])

def project(cfg):
    with Timer("wait_for_input"):
        adios_files_list = wait_for_input(cfg)
    with Timer("wait_for_model"):
        model_path = wait_for_model(cfg)

    lastN = cfg.project_lastN

    top_dir, tmp_dir, published_dir = dirs(cfg)

    dir = cfg.output_path

    with Timer("project_next"):
        cvae_input = read_lastN(adios_files_list, lastN)

    rmsds = cvae_input[1]

    with open(f'{dir}/rmsd.npy', 'wb') as ff:
        np.save(ff, rmsds)

    with Timer("project_predict"):
        embeddings_cvae = predict(cfg, model_path, cvae_input, batch_size=1024)

    with open(f'{dir}/embeddings_cvae.npy', 'wb') as ff:
        np.save(ff, embeddings_cvae)

    with Timer("project_TSNE_2D"):
        tsne2 = TSNE(n_components=2)
        tsne_embeddings2 = tsne2.fit_transform(embeddings_cvae)

    with open(f'{dir}/tsne_embeddings_2.npy', 'wb') as ff:
        np.save(ff, tsne_embeddings2)

    with Timer("project_TSNE_3D"):
        tsne3 = TSNE(n_components=3)
        tsne_embeddings3 = tsne3.fit_transform(embeddings_cvae)

    with open(f'{dir}/tsne_embeddings_3.npy', 'wb') as ff:
        np.save(ff, tsne_embeddings3)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    parser.add_argument("-p", "--project", action="store_true",
                    help="compute tsne")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = OutlierDetectionConfig.from_yaml(args.config)
    if(args.project):
        project(cfg)
    else:
        main(cfg)
