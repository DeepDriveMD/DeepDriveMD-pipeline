#!/usr/bin/env python

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

import sys
import subprocess

DDMD="/".join(os.getenv('PWD').split("/")[:-3])
PATH=os.getenv('PATH')
USER=os.getenv('USER')
LD_LIBRARY_PATH=os.getenv('LD_LIBRARY_PATH')
PYTHONPATH=os.getenv('PYTHONPATH')
PYTHON=subprocess.getstatusoutput('which python')[1]
ADIOS2="/".join(subprocess.getstatusoutput('which bpls')[1].split("/")[:-2])
CONDA="/".join(PYTHON.split("/")[:-2])

class Header(BaseModel):
    title = "smoothended_rec, aae, 12h"
    resource = "llnl.lassen"
    queue = "pbatch"
    schema_ = "local"
    project = "cv19-a01"
    walltime_min = 60 * 12
    max_iteration = 4
    cpus_per_node = 40
    gpus_per_node = 4
    hardware_threads_per_cpu = 4
    experiment_directory = f"/p/gpfs1/{USER}/Outputs/405a"
    software_directory = (
        f"{DDMD}/deepdrivemd"
    )

    init_pdb_file = "/usr/workspace/cv_ddmd/yakushin/Integration1/data/BigMolecules/smoothended_rec/system/comp.pdb"
    ref_pdb_file: Path = init_pdb_file
    config_directory = "set_by_deepdrivemd"
    adios_xml_sim = "set_by_deepdrivemd"
    adios_xml_agg = "set_by_deepdrivemd"
    adios_xml_agg_4ml = "set_by_deepdrivemd"
    adios_xml_file = "set_by_deepdrivemd"
    model = "aae"
    node_local_path: Path = "/tmp/"


header = Header()

print(yaml.dump(header.dict()))

class CPUReqMD(BaseModel):
    processes = 1
    process_type: str = None
    threads_per_process = 4
    thread_type = "OpenMP"


cpu_req_md = CPUReqMD()


class GPUReqMD(BaseModel):
    processes = 1
    process_type: str = None
    threads_per_process = 1
    thread_type = "CUDA"


gpu_req_md = GPUReqMD()


class TaskConfigMD(BaseModel):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"
    node_local_path = header.node_local_path
    pdb_file = "set_by_deepdrivemd"
    initial_pdb_dir = "/usr/workspace/cv_ddmd/yakushin/Integration1/data/BigMolecules/smoothended_rec/"
    solvent_type = "explicit"
    top_suffix: str = ".top"
    simulation_length_ns = 10.0 / 5
    report_interval_ps = 50.0 / 5
    dt_ps = 0.002
    temperature_kelvin = 300.0
    heat_bath_friction_coef = 1.0
    reference_pdb_file = f"{header.ref_pdb_file}"
    openmm_selection = ["CA"]
    mda_selection = "protein and name CA"
    threshold = 8.0
    in_memory = False
    bp_file = "set_by_deepdrivemd"
    outliers_dir = f"{header.experiment_directory}/agent_runs/stage0000/task0000/published_outliers"
    copy_velocities_p = 0.5
    next_outlier_policy = 1
    lock = "set_by_deepdrivemd"
    adios_xml_sim = header.adios_xml_sim
    adios_xml_file = header.adios_xml_file
    compute_rmsd = True
    divisibleby = 32
    zcentroid_atoms = "resname CY8 and not name H*"
    init_pdb_file = f"{header.init_pdb_file}"
    model = header.model
    compute_zcentroid = True


task_config_md = TaskConfigMD()

pre_exec_md = [
    "unset PYTHONPATH",
    "module load gcc/7.3.1",
    ". /etc/profile.d/conda.sh",
    f"conda activate {CONDA}",
    "export IBM_POWERAI_LICENSE_ACCEPT=yes",
    "module use /usr/workspace/cv_ddmd/software1/modules",
    "module load adios2/2.8.1a",
    f"export PYTHONPATH={PYTHONPATH}",
]


class MD(BaseModel):
    pre_exec = pre_exec_md
    executable = PYTHON
    arguments = [f"{header.software_directory}/sim/openmm_stream/run_openmm.py"]
    cpu_reqs = cpu_req_md.dict()
    gpu_reqs = gpu_req_md.dict()
    num_tasks = 120
    task_config = task_config_md.dict()


md = MD()


class CPUReqAgg(BaseModel):
    processes = 1
    process_type: str = None
    threads_per_process = 4 * 16
    thread_type = "OpenMP"


cpu_req_agg = CPUReqAgg()


class GPUReqAgg(BaseModel):
    processes = 0
    process_type: str = None
    threads_per_process = 0
    thread_type: str = None


gpu_req_agg = GPUReqAgg()


class TaskConfigAgg(BaseModel):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"
    node_local_path = header.node_local_path
    num_tasks = 10
    n_sim = md.num_tasks
    sleeptime_bpfiles = 30
    adios_xml_agg = header.adios_xml_agg
    adios_xml_agg_4ml = header.adios_xml_agg_4ml
    compute_rmsd = task_config_md.compute_rmsd
    model = header.model
    compute_zcentroid = task_config_md.compute_zcentroid


task_config_agg = TaskConfigAgg()


class Aggregator(BaseModel):
    pre_exec = pre_exec_md
    executable = PYTHON
    arguments = [f"{header.software_directory}/aggregation/stream/aggregator.py"]
    cpu_reqs = cpu_req_agg.dict()
    gpu_reqs = gpu_req_agg.dict()
    skip_aggregation = False
    num_tasks = task_config_agg.num_tasks
    task_config = task_config_agg.dict()


agg = Aggregator()


class AAE(BaseModel):
    latent_dim = 16
    encoder_bias = True
    encoder_relu_slope = 0.0
    encoder_filters = [64, 128, 256, 256, 512]
    encoder_kernels = [5, 5, 3, 1, 1]
    decoder_bias = True
    decoder_relu_slope = 0.0
    decoder_affine_widths = [64, 128, 512, 1024]
    discriminator_bias = True
    discriminator_relu_slope = 0.0
    discriminator_affine_widths = [512, 128, 64]
    # Mean of the prior distribution
    noise_mu = 0.0
    # Standard deviation of the prior distribution
    noise_std = 1.0
    # Releative weight to put on gradient penalty
    lambda_gp = 10.0
    # Releative weight to put on reconstruction loss
    lambda_rec: float = 0.5


class TaskConfigML(AAE):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"
    epochs = 70
    batch_size = 32
    min_step_increment = 600
    max_steps = 2000
    max_loss = 1500
    num_agg = agg.num_tasks
    timeout1 = 30
    timeout2 = 10
    agg_dir = f"{header.experiment_directory}/aggregation_runs/"
    published_model_dir = "set_by_deepdrivemd"
    checkpoint_dir = "set_by_deepdrivemd"
    adios_xml_agg = header.adios_xml_agg
    adios_xml_agg_4ml = header.adios_xml_agg_4ml
    reinit = False
    use_model_checkpoint = True
    read_batch = 2000

    # resume_checkpoint = None
    num_points: int = 459
    scalar_dset_names = []
    cms_transform: bool = True
    scalar_requires_grad = False
    split_pct = 0.8
    seed = 333
    shuffle = True
    # init_weights = None
    ae_optimizer = {"name": "Adam", "hparams": {"lr": 0.0001}}
    disc_optimizer = {"name": "Adam", "hparams": {"lr": 0.0001}}
    num_data_workers = 16
    prefetch_factor = 2
    model = header.model
    node_local_path = header.node_local_path
    init_weights_path = "/tmp"
    num_features = 0


task_config_ml = TaskConfigML()


class ML(BaseModel):
    pre_exec = pre_exec_md
    executable = PYTHON
    arguments = [f"{header.software_directory}/models/aae_stream/train.py"]
    cpu_reqs = cpu_req_md.dict()
    gpu_reqs = gpu_req_md.dict()
    task_config = task_config_ml.dict()


cpu_req_agent = cpu_req_md.copy()
cpu_req_agent.threads_per_process = 39


class TaskConfigAgent(AAE):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"

    agg_dir = f"{header.experiment_directory}/aggregation_runs"
    num_agg = agg.num_tasks
    min_step_increment = 200
    timeout1 = 30
    timeout2 = 10
    best_model = f"{header.experiment_directory}/machine_learning_runs/stage0000/task0000/published_model/best.pt"
    lastN = 1000
    outlier_count = 120
    outlier_max = 1000
    outlier_min = 120
    init_pdb_file = f"{header.init_pdb_file}"
    ref_pdb_file = f"{header.ref_pdb_file}"
    init_eps = 1.3
    init_min_samples = 10
    read_batch = 600
    num_sim = md.num_tasks
    project_lastN = 50 * 1000
    project_gpu = False
    adios_xml_agg = header.adios_xml_agg
    use_outliers = True
    use_random_outliers = True
    compute_rmsd = task_config_md.compute_rmsd
    compute_zcentroid = task_config_md.compute_zcentroid
    outlier_selection = "lof"
    model = header.model

    num_points = task_config_ml.num_points
    num_features = task_config_ml.num_features


task_config_agent = TaskConfigAgent()


class Agent(BaseModel):
    pre_exec = pre_exec_md
    executable = PYTHON
    arguments = [f"{header.software_directory}/agents/stream/dbscan.py"]
    cpu_reqs = cpu_req_agent.dict()
    gpu_reqs = gpu_req_md.dict()
    task_config = task_config_agent.dict()


class Components(BaseModel):
    molecular_dynamics_stage = MD()
    aggregation_stage = Aggregator()
    machine_learning_stage = ML()
    agent_stage = Agent()


components = Components()
print(yaml.dump(components.dict()))
