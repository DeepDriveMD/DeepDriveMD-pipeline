import os
import shutil
from pathlib import Path
from typing import Optional

import radical.utils as ru  # type: ignore[import]
from radical.entk import AppManager, Pipeline, Stage, Task  # type: ignore[import]
from deepdrivemd.config import BaseSettings, BaseStageConfig
from deepdrivemd.utils import parse_args

from md import OpenMMSimulationConfig
from utils import OpenMMSimulationParameters


class OpenMMEnsembleConfig(BaseSettings):
    # Radical parameters
    title: str
    resource: str
    queue: str
    schema_: str
    project: str
    walltime_min: int
    cpus_per_node: int
    gpus_per_node: int
    hardware_threads_per_cpu: int

    # Ensemble parameters
    output_directory: str
    input_pdb_file: str
    input_top_file: Optional[str] = None
    num_tasks: int = 120

    # Task parameters (hardware requirements, executable)
    task_config: BaseStageConfig = BaseStageConfig()

    # Individual simulation parameters
    sim_params: OpenMMSimulationParameters = OpenMMSimulationParameters()


def generate_task(cfg: BaseStageConfig) -> Task:
    task = Task()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    task.pre_exec = cfg.pre_exec.copy()
    task.executable = cfg.executable
    task.arguments = cfg.arguments.copy()
    return task


def generate_pipeline(cfg: OpenMMEnsembleConfig) -> Pipeline:
    pipeline = Pipeline()
    stage = Stage()
    for task_id in range(cfg.num_tasks):
        # Create an output folder
        task_output_dir = Path(cfg.output_directory) / f"task-{task_id:04d}"
        task_output_dir.mkdir()

        # Setup task config
        simulation_config = OpenMMSimulationConfig(
            input_pdb_file=cfg.input_pdb_file,
            input_top_file=cfg.input_top_file,
            output_traj_file=str(task_output_dir / "traj.dcd"),
            output_log_file=str(task_output_dir / "traj.log"),
            params=cfg.sim_params,
        )
        # Assign unique random seed to each task
        simulation_config.params.random_seed = task_id

        # Write simulation config as a YAML file
        cfg_path = task_output_dir / "config.yaml"
        simulation_config.dump_yaml(cfg_path)

        # Create entk task to run the ensemble member
        task = generate_task(cfg.task_config)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

    pipeline.add_stages(stage)
    return pipeline


def main(cfg: OpenMMEnsembleConfig) -> None:

    reporter = ru.Reporter(name="radical.entk")
    reporter.title(cfg.title)

    # Create Application Manager
    try:
        appman = AppManager(
            hostname=os.environ["RMQ_HOSTNAME"],
            port=int(os.environ["RMQ_PORT"]),
            username=os.environ["RMQ_USERNAME"],
            password=os.environ["RMQ_PASSWORD"],
        )
    except KeyError:
        raise ValueError(
            "Invalid RMQ environment. Please see README.md for configuring environment."
        )

    # Calculate total number of nodes required. Assumes 1 MD job per GPU
    num_full_nodes, extra_gpus = divmod(cfg.num_tasks, cfg.gpus_per_node)
    extra_node = int(extra_gpus > 0)
    num_nodes = max(1, num_full_nodes + extra_node)

    appman.resource_desc = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus_per_node * cfg.hardware_threads_per_cpu * num_nodes,
        "gpus": cfg.gpus_per_node * num_nodes,
    }

    pipeline = generate_pipeline(cfg)
    # Assign the workflow as a list of Pipelines to the Application Manager.
    # All the pipelines in the list will execute concurrently.
    appman.workflow = [pipeline]

    # Run the Application Manager
    appman.run()


if __name__ == "__main__":
    args = parse_args()
    cfg = OpenMMEnsembleConfig.from_yaml(args.config)
    # Backup configuration file
    Path(cfg.output_directory).mkdir()
    shutil.copy(args.config, cfg.output_directory)
    # Launch entk application
    main(cfg)
