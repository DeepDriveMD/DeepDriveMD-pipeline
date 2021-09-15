#!/usr/bin/env python

import yaml
from pydantic import BaseModel
import os
from pathlib import Path


class Header(BaseModel):
    title = "BBA integration test"
    resource = "llnl.lassen"
    queue = "pbatch"
    schema_ = "local"
    project = "cv19-a01"
    walltime_min = 60 * 3
    max_iteration = 4
    cpus_per_node = 40
    gpus_per_node = 4
    hardware_threads_per_cpu = 4
    experiment_directory = "/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/302"
    software_directory = (
        "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd"
    )
    node_local_path: Path = None
    init_pdb_file = "set_by_deepdrivemd"
    ref_pdb_file: Path = None
    config_directory = "set_by_deepdrivemd"
    adios_xml_sim = "set_by_deepdrivemd"
    adios_xml_agg = "set_by_deepdrivemd"
    multi_ligand_table = (
        "/usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/ml_table.csv"
    )


header = Header()

print(yaml.dump(header.dict()))

pythonpath = os.getenv("PYTHONPATH")
python = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"


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
    node_local_path = "set_by_deepdrivemd"
    pdb_file = "set_by_deepdrivemd"
    initial_pdb_dir = "set_by_deepdrivemd"
    solvent_type = "explicit"
    top_suffix: str = ".prmtop"
    simulation_length_ns = 10.0 / 10
    report_interval_ps = 50.0 / 10
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
    compute_rmsd = False
    divisibleby = 32
    multi_ligand_table = header.multi_ligand_table


task_config_md = TaskConfigMD()

pre_exec_md = [
    "unset PYTHONPATH",
    "module load gcc/7.3.1",
    ". /etc/profile.d/conda.sh",
    "conda activate /usr/workspace/cv_ddmd/conda1/powerai",
    "export IBM_POWERAI_LICENSE_ACCEPT=yes",
    "module use /usr/workspace/cv_ddmd/software1/modules",
    "module load adios2",
    f"export PYTHONPATH={pythonpath}",
]


class MD(BaseModel):
    pre_exec = pre_exec_md
    executable = python
    arguments = [f"{header.software_directory}/sim/openmm_stream/run_openmm.py"]
    cpu_reqs = cpu_req_md.dict()
    gpu_reqs = gpu_req_md.dict()
    num_tasks = 12
    task_config = task_config_md.dict()


md = MD()


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
    node_local_path = "set_by_deepdrivemd"
    num_tasks = 1
    n_sim = md.num_tasks
    sleeptime_bpfiles = 30
    adios_xml_agg = header.adios_xml_agg
    compute_rmsd = task_config_md.compute_rmsd
    multi_ligand_table = header.multi_ligand_table


task_config_agg = TaskConfigAgg()


class Aggregator(BaseModel):
    pre_exec = pre_exec_md
    executable = python
    arguments = [f"{header.software_directory}/aggregation/stream/aggregator.py"]
    cpu_reqs = cpu_req_md.dict()
    gpu_reqs = gpu_req_agg.dict()
    skip_aggregation = False
    num_tasks = task_config_agg.num_tasks
    task_config = task_config_agg.dict()


agg = Aggregator()


class CVAE(BaseModel):
    initial_shape = [288, 288]
    final_shape = [288, 288, 1]
    split_pct = 0.8
    shuffle = True
    latent_dim = 10
    conv_layers = 4
    conv_filters = [64] * 4
    conv_filter_shapes = [[3, 3]] * 4
    conv_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0.4]


class TaskConfigML(CVAE):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"
    epochs = 200
    batch_size = 32
    min_step_increment = 200
    max_steps = 2000
    max_loss = 1500
    num_agg = agg.num_tasks
    timeout1 = 30
    timeout2 = 10
    agg_dir = f"{header.experiment_directory}/aggregation_runs/"
    published_model_dir = "set_by_deepdrivemd"
    checkpoint_dir = "set_by_deepdrivemd"
    adios_xml_agg = header.adios_xml_agg
    reinit = True
    use_model_checkpoint = True
    read_batch = 2000


task_config_ml = TaskConfigML()


class ML(BaseModel):
    pre_exec = pre_exec_md
    executable = python
    arguments = [f"{header.software_directory}/models/keras_cvae_stream/train.py"]
    cpu_reqs = cpu_req_md.dict()
    gpu_reqs = gpu_req_md.dict()
    task_config = task_config_ml.dict()


cpu_req_agent = cpu_req_md.copy()
cpu_req_agent.threads_per_process = 39


class TaskConfigAgent(CVAE):
    experiment_directory = "set_by_deepdrivemd"
    stage_idx = 0
    task_idx = 0
    output_path = "set_by_deepdrivemd"

    agg_dir = f"{header.experiment_directory}/aggregation_runs"
    num_agg = agg.num_tasks
    min_step_increment = 500
    timeout1 = 30
    timeout2 = 10
    best_model = f"{header.experiment_directory}/machine_learning_runs/stage0000/task0000/published_model/best.h5"
    lastN = 2000
    outlier_count = 120
    outlier_max = 5000
    outlier_min = 100
    init_pdb_file = f"{header.init_pdb_file}"
    ref_pdb_file = f"{header.ref_pdb_file}"
    init_eps = 1.3
    init_min_samples = 10
    read_batch = 2000
    num_sim = md.num_tasks
    project_lastN = 50 * 1000
    project_gpu = False
    adios_xml_agg = header.adios_xml_agg
    use_outliers = True
    use_random_outliers = True
    compute_rmsd = task_config_md.compute_rmsd
    multi_ligand_table = header.multi_ligand_table


task_config_agent = TaskConfigAgent()


class Agent(BaseModel):
    pre_exec = pre_exec_md
    executable = python
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
