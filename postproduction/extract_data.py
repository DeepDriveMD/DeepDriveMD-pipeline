#!/usr/bin/env python

import numpy
import h5py
import tensorflow as tf
import argparse
from pathlib import Path
from deepdrivemd.models.keras_cvae.config import KerasCVAEModelConfig
from deepdrivemd.models.keras_cvae.model import CVAE
from deepdrivemd.models.keras_cvae.utils import sparse_to_dense

def parse_args():
    '''
    Parse the command line arguments

    The relevant information is provided in the source code.
    '''
    describe = '''
Extract_data pulls data from the HDF5 files that DeepDriveMD generated
and stores it in a format suitable for visualization with the Matplotlib
library.'''
    example = '''
example:

   ./extract_data.py \\
           ./machine_learning_runs/stage0008/task0000/stage0008_task0000.yaml \\
           ./machine_learning_runs/stage0008/task0000/checkpoint/epoch-50-20231108-132138.h5 \\
           ./molecular_dynamics_runs "4,1" data_points.csv
    '''
    parser = argparse.ArgumentParser(description=describe,
                                     epilog=example,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",help="The YAML file containing the machine learning model configuration")
    parser.add_argument("model",help="The file containing the HDF5 machine learning model")
    parser.add_argument("data_path",help="The path below which the HFD5 time step files live")
    parser.add_argument("latent_dims",help="The 2 or 3 latent space dimensions to plot given as a comma separated list of 1-based integers")
    parser.add_argument("file_out",help="Output filename")
    return parser.parse_args()

def extract_from_data_file(fp_out,cfg,file,model,ndim,dims):
    '''
    Extract data from the file and store the results in the output file

    - open the HDF5 file
    - loop over all frames:
      - extract contact_map
      - extract RMSD
      - encode contact_map
      - extract the selected dimensions from the latent space
      - store latent space coordinates and RMSD in output file
    - close HDF5 file
    '''
    print("file: "+str(file)+"\n")
    data = sparse_to_dense(
        file, cfg.dataset_name, cfg.initial_shape, cfg.final_shape
    )
    latent = model.return_embeddings(data)
    with h5py.File(file,"r") as h5_file:
        rmsd = numpy.array(h5_file["rmsd"][:])
    for ii in range(rmsd.shape[0]):
        value = rmsd[ii]
        xx = latent[ii,int(dims[0])]
        yy = latent[ii,int(dims[1])]
        fp_out.write(f'{xx}, {yy}, ')
        if ndim == 3:
            zz = latent[ii,int(dims[2])]
            fp_out.write(f'{zz}, ')
        fp_out.write(f'{value}\n')
    return

def extract_from_data_files(fp_out,cfg,data_files,model,ndim,dims):
    '''
    Extract data from each of the data files and write the relevant information
    to fp_out.
    
    - data_files: a list of data files
    - model: the autoencoder to convert data to the latent space coordinates
    - ndim: the number of dimensions to extract from the latent space
    - dims: the dimensions to extract from the latent (list of 0-based integers)
    '''
    for file in data_files:
        extract_from_data_file(fp_out,cfg,file,model,ndim,dims)

def extract_data(args):
    config_file = Path(args.config)
    model_file  = Path(args.model)
    data_path   = Path(args.data_path)
    data_files  = list(data_path.glob('**/*.h5'))
    out_file    = Path(args.file_out)
    dims        = str(args.latent_dims).split(",")
    ndim        = len(dims)
    if ndim < 2:
        print("This tool generates data to plot functions of 2 or 3 dimensions")
        print("you have specified only "+str(ndim)+" dimensions: "+str(args.latent_dims))
        sys.exit(1)
    if ndim > 3:
        print("This tool generates data to plot functions of 2 or 3 dimensions")
        print("you have specified "+str(ndim)+" dimensions: "+str(args.latent_dims))
        sys.exit(1)
    #model = tf.keras.saving.load_model(model_file) # as of TF 2.12
    #DEBUG
    print(model_file)
    #DEBUG
    cfg = KerasCVAEModelConfig.from_yaml(config_file) # before TF 2.12 (to be deleted)
    cvae = CVAE(
            image_size=cfg.final_shape[:2],
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
    cvae.model.load_weights(model_file)
    #model = tf.keras.models.load_model(model_file) # before TF 2.12 (to be deleted)
    # model.encoder(x) encodes input x and returns the latent space coordinates
    fp_out = open(out_file,"w")
    extract_from_data_files(fp_out,cfg,data_files,cvae,ndim,dims)
    fp_out.close()
    pass

if __name__ == "__main__":
    args = parse_args()
    extract_data(args)
