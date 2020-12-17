import os
import sys
from itertools import cycle
from pathlib import Path
from typing import List

import radical.utils as ru
from radical.entk import AppManager, Pipeline, Stage, Task

from deepdrivemd.config import ExperimentConfig
from deepdrivemd.data.api import DeepDriveMD_API


def get_initial_pdbs(initial_pdb_dir: Path) -> List[Path]:
    """Scan input directory for PDBs and optional topologies."""

    pdb_filenames = list(initial_pdb_dir.glob("*/*.pdb"))

    if any("__" in filename.as_posix() for filename in pdb_filenames):
        raise ValueError("Initial PDB files cannot contain a double underscore __")

    return pdb_filenames


class PipelineManager:

    PIPELINE_NAME = "DeepDriveMD"
    MD_STAGE_NAME = "MD"
    AGGREGATION_STAGE_NAME = "aggregating"
    ML_STAGE_NAME = "learning"
    AGENT_STAGE_NAME = "agent"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cur_iteration = 0

        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self.pipeline = Pipeline()
        pipeline.name = self.PIPELINE_NAME

        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.md_dir.mkdir()
        self.api.aggregation_dir.mkdir()
        self.api.ml_dir.mkdir()
        self.api.agent_dir.mkdir()
        self.api.tmp_dir.mkdir()

    # TODO: move all these paths into DeepDriveMD_API

    def aggregated_data_path(self, iteration: int) -> Path:
        return self.api.aggregation_dir.joinpath(f"data_{iteration:03d}.h5")

    def model_path(self, iteration: int) -> Path:
        return self.api.ml_dir.joinpath(f"model_{iteration:03d}")

    def latest_ml_checkpoint_path(self, iteration: int) -> Path:
        # TODO: this code requires specific checkpoint file format
        #       might want an interface class to implement a latest_checkpoint
        #       function.
        checkpoint_files = (
            self.model_path(iteration).joinpath("checkpoint").glob("*.pt")
        )
        # Format: epoch-1-20200922-131947.pt
        return max(checkpoint_files, key=lambda x: x.as_posix().split("-")[1])

    def outlier_pdbs_path(self, iteration: int) -> Path:
        return self.api.agent_dir.joinpath(f"outlier_pdbs_{iteration:03d}")

    def aggregation_config_path(self, iteration: int) -> Path:
        return self.api.aggregation_dir.joinpath(f"aggregation_{iteration:03d}.yaml")

    def ml_config_path(self, iteration: int) -> Path:
        return self.api.ml_dir.joinpath(f"ml_{iteration:03d}.yaml")

    def agent_config_path(self, iteration: int) -> Path:
        return self.api.agent_dir.joinpath(f"od_{iteration:03d}.yaml")

    def func_condition(self):
        if self.cur_iteration < self.cfg.max_iteration:
            self.func_on_true()
        else:
            self.func_on_false()

    def func_on_true(self):
        print(f"Finishing stage {self.cur_iteration} of {self.cfg.max_iteration}")
        self._generate_pipeline_iteration()

    def func_on_false(self):
        print("Done")

    def _generate_pipeline_iteration(self):

        self.pipeline.add_stages(self.generate_md_stage())

        if not cfg.aggregation_stage.skip_aggregation:
            self.pipeline.add_stages(self.generate_aggregating_stage())

        if self.cur_iteration % cfg.ml_stage.retrain_freq == 0:
            self.pipeline.add_stages(self.generate_ml_stage())

        agent_stage = self.generate_agent_stage()
        agent_stage.post_exec = self.func_condition
        self.pipeline.add_stages(agent_stage)

        self.cur_iteration += 1

    def generate_pipeline(self) -> Pipeline:
        self._generate_pipeline_iteration()
        return self.pipeline

    def generate_md_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MD_STAGE_NAME
        cfg = self.cfg.md_stage

        if self.cur_iteration > 0:
            filenames = [self.api.get_restart_points_path(self.cur_iteration - 1)]
        else:
            filenames = get_initial_pdbs(cfg.run_config.initial_pdb_dir)

        for i, filename in zip(range(cfg.num_jobs), cycle(filenames)):
            task = Task()
            task.cpu_reqs = cfg.cpu_reqs.dict()
            task.gpu_reqs = cfg.gpu_reqs.dict()
            task.pre_exec = cfg.pre_exec
            task.executable = cfg.executable
            task.arguments = cfg.arguments

            # Set unique output directory name for task
            dir_prefix = f"run{self.cur_iteration:03d}_{i:04d}"

            # Update base parameters
            cfg.run_config.experiment_directory = self.cfg.experiment_directory
            cfg.run_config.result_dir = self.api.md_dir
            cfg.run_config.dir_prefix = dir_prefix
            if self.cur_iteration > 0:
                cfg.restart_point = i
            else:
                cfg.run_config.pdb_file = filename

            # Write MD yaml to tmp directory to be picked up and moved by MD job
            cfg_path = self.api.tmp_dir.joinpath(f"{dir_prefix}.yaml")
            cfg.run_config.dump_yaml(cfg_path)
            task.arguments += ["-c", cfg_path]
            stage.add_tasks(task)

        return stage

    def generate_aggregating_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGGREGATION_STAGE_NAME
        cfg = self.cfg.aggregation_stage

        task = Task()
        task.cpu_reqs = cfg.cpu_reqs.dict()
        task.pre_exec = cfg.pre_exec
        task.executable = cfg.executable
        task.arguments = cfg.arguments

        # Update base parameters
        cfg.run_config.experiment_directory = self.cfg.experiment_directory
        cfg.run_config.output_path = self.aggregated_data_path(self.cur_iteration)

        # Write yaml configuration
        cfg_path = self.aggregation_config_path(self.cur_iteration)
        cfg.run_config.dump_yaml(cfg_path)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_ml_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.ML_STAGE_NAME
        cfg = self.cfg.ml_stage

        task = Task()
        task.cpu_reqs = cfg.cpu_reqs.dict()
        task.gpu_reqs = cfg.gpu_reqs.dict()
        task.pre_exec = cfg.pre_exec
        task.executable = cfg.executable
        task.arguments = cfg.arguments

        # Update base parameters
        cfg.run_config.experiment_directory = self.cfg.experiment_directory
        cfg.run_config.input_path = self.aggregated_data_path(self.cur_iteration)
        cfg.run_config.output_path = self.model_path(self.cur_iteration)
        if self.cur_iteration > 0:
            cfg.run_config.init_weights_path = self.latest_ml_checkpoint_path(
                self.cur_iteration - 1
            )

        # Write yaml configuration
        cfg_path = self.ml_config_path(self.cur_iteration)
        cfg.run_config.dump_yaml(cfg_path)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGENT_STAGE_NAME
        cfg = self.cfg.agent_stage

        task = Task()
        task.cpu_reqs = cfg.cpu_reqs.dict()
        task.gpu_reqs = cfg.gpu_reqs.dict()
        task.pre_exec = cfg.pre_exec
        task.executable = cfg.executable
        task.arguments = cfg.arguments

        self.outlier_pdbs_path(self.cur_iteration).mkdir()

        # Update base parameters
        cfg.run_config.experiment_directory = self.cfg.experiment_directory
        cfg.run_config.input_path = self.aggregated_data_path(self.cur_iteration)
        cfg.run_config.model_path = self.ml_config_path(self.cur_iteration)
        cfg.run_config.output_path = self.outlier_pdbs_path(self.cur_iteration)
        cfg.run_config.weights_path = self.latest_ml_checkpoint_path(self.cur_iteration)

        # Write yaml configuration
        cfg_path = self.agent_config_path(self.cur_iteration)
        cfg.run_config.dump_yaml(cfg_path)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage


if __name__ == "__main__":

    # Read YAML configuration file from stdin
    try:
        config_filename = sys.argv[1]
    except Exception:
        raise ValueError(f"Usage:\tpython {sys.argv[0]} [config.json]\n\n")

    cfg = ExperimentConfig.from_yaml(config_filename)

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
    num_nodes = max(1, cfg.md_stage.num_jobs // cfg.gpus_per_node)

    appman.resource_desc = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus_per_node * cfg.hardware_threads_per_cpu * num_nodes,
        "gpus": cfg.gpus_per_node * num_nodes,
    }

    pipeline_manager = PipelineManager(cfg)
    pipeline = pipeline_manager.generate_pipeline()

    # Assign the workflow as a list of Pipelines to the Application Manager. In
    # this way, all the pipelines in the list will execute concurrently.
    appman.workflow = [pipeline]

    # Run the Application Manager
    appman.run()
