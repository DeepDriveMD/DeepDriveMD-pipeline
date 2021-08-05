#!/usr/bin/env python
import os
from pathlib import Path
from deepdrivemd.config import (
    CPUReqs,
    GPUReqs,
    StreamingExperimentConfig,
    MolecularDynamicsStageConfig,
    StreamingAggregationStageConfig,
    MachineLearningStageConfig,
    AgentStageConfig,
)
from deepdrivemd.sim.openmm_stream.config import OpenMMConfig
from deepdrivemd.aggregation.stream.config import StreamAggregation
from deepdrivemd.models.keras_cvae_stream.config import KerasCVAEModelConfig
from deepdrivemd.agents.stream.config import OutlierDetectionConfig


# TODO: This file should have no "set_by_deepdrivemd" occurences.
#       Instead, the config classes should have it by default.

software_directory = (
    "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd"
)

# TODO: we no longer have a misc path, should update this (we probably don't need it anymore)
pythonpath = f"{software_directory}/misc/:{os.getenv('PYTHONPATH')}"
python = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"

experiment_directory = Path("/usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/14")
init_pdb_file = Path(
    "/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_input/1FME-0.pdb"
)
ref_pdb_file = Path(
    "/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_reference/1FME.pdb"
)

pre_exec = [
    "unset PYTHONPATH",
    "module load gcc/7.3.1",
    ". /etc/profile.d/conda.sh",
    "conda activate /usr/workspace/cv_ddmd/conda1/powerai",
    "export IBM_POWERAI_LICENSE_ACCEPT=yes",
    "module use /usr/workspace/cv_ddmd/software1/modules",
    "module load adios2",
    f"export PYTHONPATH={pythonpath}",
]

sim_tasks = 120
agg_tasks = 10


cpu_req_md = CPUReqs(
    processes=1, process_type=None, threads_per_process=4, thread_type="OpenMP"
)


gpu_req_md = GPUReqs(
    processes=1, process_type=None, threads_per_process=1, thread_type="CUDA"
)


gpu_req_agg = GPUReqs(
    processes=0, process_type=None, threads_per_process=0, thread_type=None
)

cpu_req_agent = CPUReqs(
    processes=1, process_type=None, threads_per_process=39, thread_type="OpenMP"
)

sim_cfg = OpenMMConfig(
    initial_pdb_dir="/usr/workspace/cv_ddmd/yakushin/Integration1/data/bba/ddmd_input",
    solvent_type="implicit",
    top_suffix=None,
    simulation_length_ns=10.0,
    report_interval_ps=50.0,
    dt_ps=0.002,
    temperature_kelvin=300.0,
    heat_bath_friction_coef=1.0,
    reference_pdb_file=ref_pdb_file,
    openmm_selection=["CA"],
    mda_selection="protein and name CA",
    threshold=8.0,
    in_memory=False,
    outliers_dir=experiment_directory.joinpath(
        "/agent_runs/stage0000/task0000/published_outliers"
    ),
    copy_velocities_p=0.5,
    next_outlier_policy=1,
    adios_xml_sim="set_by_deepdrivemd",
)

# TODO: Give adios_xml_agg a default value in the StreamAggregation, OpenMMConfig
# and  KerasCVAEModelConfig definitions
agg_cfg = StreamAggregation(
    n_sim=sim_tasks,
    sleeptime_bpfiles=30,
    num_tasks=2,
    adios_xml_agg=Path("set_by_deepdrivemd"),
)

ml_cfg = KerasCVAEModelConfig(
    initial_shape=(28, 28),
    final_shape=[28, 28, 1],
    split_pct=0.8,
    shuffle=True,
    latent_dim=10,
    conv_layers=4,
    conv_filters=[64] * 4,
    conv_filter_shapes=[(3, 3)] * 4,
    conv_strides=[(1, 1), (2, 2), (1, 1), (1, 1)],
    dense_layers=1,
    dense_neurons=[128],
    dense_dropouts=[0.4],
    epochs=50,
    batch_size=32,
    min_step_increment=1000,
    max_steps=2000,
    max_loss=100,
    num_agg=agg_tasks,
    timeout1=30,
    timeout2=10,
    agg_dir=experiment_directory / "aggregation_runs/",
    published_model_dir="set_by_deepdrivemd",
    checkpoint_dir="set_by_deepdrivemd",
    adios_xml_agg="set_by_deepdrivemd",
    reinit=True,
    use_model_checkpoint=True,
    read_batch=2000,
)

agent_cfg = OutlierDetectionConfig(
    agg_dir=experiment_directory / "aggregation_runs",
    num_agg=agg_tasks,
    min_step_increment=500,
    timeout1=30,
    timeout2=10,
    best_model=experiment_directory.joinpath(
        "machine_learning_runs/stage0000/task0000/published_model/best.h5"
    ),
    lastN=2000,
    outlier_count=120,
    outlier_max=5000,
    outlier_min=1000,
    init_pdb_file=init_pdb_file,
    ref_pdb_file=ref_pdb_file,
    init_eps=1.3,
    init_min_samples=10,
    read_batch=2000,
    num_sim=sim_tasks,
    project_lastN=50 * 1000,
    project_gpu=False,
    adios_xml_agg="set_by_deepdrivemd",
    use_outliers=True,
)

config = StreamingExperimentConfig(
    title="BBA integration test",
    resource="llnl.lassen",
    queue="pbatch",
    schema_="local",
    project="cv19-a01",
    walltime_min=60 * 12,
    max_iteration=4,
    cpus_per_node=40,
    gpus_per_node=4,
    hardware_threads_per_cpu=4,
    experiment_directory=experiment_directory,
    node_local_path=None,
    molecular_dynamics_stage=MolecularDynamicsStageConfig(
        pre_exec=pre_exec,
        executable=python,
        arguments=[f"{software_directory}/sim/openmm_stream/run_openmm.py"],
        cpu_reqs=cpu_req_md,
        gpu_reqs=gpu_req_md,
        num_tasks=sim_tasks,
        task_config=sim_cfg,
    ),
    model_selection_stage=None,
    aggregation_stage=StreamingAggregationStageConfig(
        pre_exec=pre_exec,
        executable=python,
        arguments=[f"{software_directory}/aggregation/stream/aggregator.py"],
        cpu_reqs=cpu_req_md,
        gpu_reqs=gpu_req_agg,
        num_tasks=agg_tasks,
        skip_aggregation=False,
        task_config=agg_cfg,
    ),
    machine_learning_stage=MachineLearningStageConfig(
        pre_exec=pre_exec,
        executable=python,
        arguments=[f"{software_directory}/models/keras_cvae_stream/train.py"],
        cpu_reqs=cpu_req_md,
        gpu_reqs=gpu_req_md,
        task_config=ml_cfg,
    ),
    agent_stage=AgentStageConfig(
        pre_exec=pre_exec,
        executable=python,
        arguments=[f"{software_directory}/agents/stream/dbscan.py"],
        cpu_reqs=cpu_req_agent,
        gpu_reqs=gpu_req_md,
        task_config=agent_cfg,
    ),
    adios_xml_sim="set_by_deepdrivemd",
    adios_xml_agg="set_by_deepdrivemd",
    config_directory="set_by_deepdrivemd",
    init_pdb_file=init_pdb_file,
    ref_pdb_file=ref_pdb_file,
)

config.dump_yaml("test.yaml")
