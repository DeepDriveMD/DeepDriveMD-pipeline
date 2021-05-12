import os
import shutil
import radical.utils as ru
from radical.entk import AppManager
from deepdrivemd.utils import parse_args
from deepdrivemd.config import ExperimentConfig
from deepdrivemd.workflow.sync import SyncPipelineManager


if __name__ == "__main__":

    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

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
    # TODO: fix this assumption for NAMD
    num_full_nodes, extra_gpus = divmod(
        cfg.molecular_dynamics_stage.num_tasks, cfg.gpus_per_node
    )
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

    if cfg.workflow_mode == "synchronous":
        pipeline_manager = SyncPipelineManager(cfg)
    else:
        raise ValueError(f"Invalid workflow_mode: {cfg.workflow_mode}")

    # Back up configuration file (PipelineManager must create cfg.experiment_dir)
    shutil.copy(args.config, cfg.experiment_directory)

    pipelines = pipeline_manager.generate_pipelines()
    # Assign the workflow as a list of Pipelines to the Application Manager.
    # All the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
