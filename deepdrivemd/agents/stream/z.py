import os 
import numpy as np
import h5py 
import errno 
import MDAnalysis as mda 
from cvae.CVAE import CVAE
#from keras import backend as K
import tensorflow.keras.backend as K
#from sklearn.cluster import DBSCAN 
from cuml import DBSCAN as DBSCAN
import cuml
import cupy as cp
import numpy as np


import adios2

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full

def read_adios_file(adios_file):
    cms = []
    positions = []
    f = adios2.open(adios_file, mode='r')
    for fs in f:
        cm = fs.read('contact_map')
        position = fs.read('positions')
        cms.append(cm)
        positions.append(position)
    f.close()
    return cms, positions

'''
def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 

def adios_cm_to_cvae(adios_cm_data_lists): 
    cm_all = np.hstack(adios_cm_data_lists)
    big_cms = []
    big_positions = []
    for fn in cm_all:
        small = read_adios_file(fn)
        big_cms += small[0]
        big_positions += small[1]
    return big_cms, big_positions
'''


def adios_to_cvae(fn, lastN=2000):
    with adios2.open(fn, "r") as fr:
        n = fr.steps()
        vars = fr.available_variables()
        # print("vars = ", vars)
        print("n = ", n)
        results = {}
        for v in ['contact_map', 'positions', 'md5', 'velocities']:
            print(v)
            if(v != 'md5'):
                shape = list(map(int, vars[v]['Shape'].split(",")))
                zs = list(np.zeros(len(shape), dtype='int'))
                results[v] = fr.read(v, zs, shape, 0, n)
                # print(results[v].shape)
            else:
                results[v] = fr.read_string(v,0,n)
    
    return results['contact_map'][-lastN:], results['positions'][-lastN:], results['md5'][-lastN:], results['velocities'][-lastN:]


def adios_to_cvae_var(fn, v, lastN=2000):
    with adios2.open(fn, "r") as fr:
        n = fr.steps()
        print("n = ", n)
        shape = list(map(int, vars[v]['Shape'].split(",")))
        zs = list(np.zeros(len(shape), dtype='int'))
        result = fr.read(v, zs, shape, 0, n)    
    return result[-lastN:]

def adios_list_to_cvae(fl, lastN=2000):
    cm = []
    positions = []
    md5 = []
    velocities = []
    for fn in fl:
        t = adios_to_cvae(fn, lastN)
        cm.append(t[0])
        positions.append(t[1])
        md5.append(t[2])
        velocities.append(t[3])
    return np.concatenate(cm), np.concatenate(positions), np.concatenate(md5), np.concatenate(velocities)


def adios_list_to_cvae_var(fl, v, lastN=2000):
    results = []
    for fn in fl:
        t = adios_to_cvae_var(fn, v, lastN)
        results.append(t)
    return np.concatenate(results)


'''

def adios_to_cvae(adios_file_lists):
    big_cm = []
    big_positions = []
    for fn in adios_file_lists:
        small = read_adios_file(fn)
        cms = small[0]
        cms2 = [triu_to_full(cm_data) for cm_data in cms]
        big_cm += cms2
        big_positions += small[1]

    cm_data_full = np.array(big_cm)

    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input, big_positions
'''


'''
def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input
'''

def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
    
'''
def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[key]) < 0: 
            dir_name = os.path.dirname(key) 
            traj_file = os.path.join(dir_name, 'output.dcd')             
            return traj_file, local_frame
        else: 
            local_frame -= int(traj_dict[key])
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb
'''

def write_pdb_frame(frame, original_pdb, output_pdb_fn):
    pdb = PDBFile(original_pdb)
    f = open(output_pdb_fn, 'w')
    PDBFile.writeFile(pdb.getTopology(), frame, f)
    f.close()


def make_dir_p(path_name): 
    try:
        os.mkdir(path_name)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def outliers_from_cvae(model_weight, cvae_input, hyper_dim=3, eps=0.35): 
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(0)  
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cvae.model.load_weights(model_weight)
    cm_predict = cvae.return_embeddings(cvae_input) 
    print("In outliers_from_cvae")
    print("cm_predict")
    print(cm_predict)
    db = DBSCAN(eps=eps, min_samples=10).fit(cm_predict)
    db_label = db.labels_
    print("db_label")
    print(list(db_label))
    outlier_list = np.where(db_label == -1)
    K.clear_session()
    return outlier_list

def predict_from_cvae(model_weight, cvae_input, hyper_dim=3): 
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(0)  
    print("predict 1:", type(cvae_input))
    print("predict 1:", cvae_input.shape)
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cvae.model.load_weights(model_weight)
    cm_predict = cvae.return_embeddings(cvae_input) 
    del cvae 
    K.clear_session()
    return cm_predict

def outliers_from_latent(cm_predict, eps=0.35):
    #print(f"In outliers_from_latent eps={eps}")
    #print("cm_predict")
    #print(type(cm_predict))
    #print(cm_predict) 
    cm_predict = cp.asarray(cm_predict)
    #print(type(cm_predict))
    #print(cm_predict)
    db = DBSCAN(eps=eps, min_samples=10, max_mbytes_per_batch=500).fit(cm_predict)
    #print(dir(db))
    db_label = db.labels_.to_array()
    #print("db_label")
    #print(list(db_label))
    outlier_list = np.where(db_label == -1)
    return outlier_list
