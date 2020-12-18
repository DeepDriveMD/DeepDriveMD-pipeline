import os
import sys
from itertools import cycle
from typing import List
import radical.utils as ru
from radical.entk import AppManager, Pipeline, Stage, Task
from deepdrivemd.config import ExperimentConfig, BaseStageConfig
from deepdrivemd.data.api import DeepDriveMD_API


def generate_task(cfg: BaseStageConfig) -> Task:
    task = Task()
    task.cpu_reqs = cfg.cpu_reqs.dict()
    task.gpu_reqs = cfg.gpu_reqs.dict()
    task.pre_exec = cfg.pre_exec
    task.executable = cfg.executable
    task.arguments = cfg.arguments
    return task


class PipelineManager:

    PIPELINE_NAME = "DeepDriveMD"
    MOLECULAR_DYNAMICS_STAGE_NAME = "MolecularDynamics"
    AGGREGATION_STAGE_NAME = "Aggregating"
    MACHINE_LEARNING_STAGE_NAME = "Learning"
    MODEL_SELECTION_STAGE_NAME = "ModelSelection"
    AGENT_STAGE_NAME = "Agent"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cur_iteration = 0

        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self.pipeline = Pipeline()
        self.pipeline.name = self.PIPELINE_NAME

        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_dir.mkdir()
        self.api.aggregation_dir.mkdir()
        self.api.machine_learning_dir.mkdir()
        self.api.model_selection_dir.mkdir()
        self.api.agent_dir.mkdir()
        self.api.tmp_dir.mkdir()

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

        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())

        if not cfg.aggregation_stage.skip_aggregation:
            self.pipeline.add_stages(self.generate_aggregating_stage())

        if self.cur_iteration % cfg.machine_learning_stage.retrain_freq == 0:
            self.pipeline.add_stages(self.generate_machine_learning_stage())
            self.pipeline.add_stages(self.generate_model_selection_stage())

        agent_stage = self.generate_agent_stage()
        agent_stage.post_exec = self.func_condition
        self.pipeline.add_stages(agent_stage)

        self.cur_iteration += 1

    def generate_pipelines(self) -> List[Pipeline]:
        self._generate_pipeline_iteration()
        return [self.pipeline]

    def generate_molecular_dynamics_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MOLECULAR_DYNAMICS_STAGE_NAME
        cfg = self.cfg.molecular_dynamics_stage

        if self.cur_iteration > 0:
            filenames = [self.api.get_agent_json_path(self.cur_iteration - 1)]
        else:
            filenames = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)

        for i, filename in zip(range(cfg.num_jobs), cycle(filenames)):

            # Set unique output directory name for task
            dir_prefix = f"run{self.cur_iteration:03d}_{i:04d}"

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.result_dir = self.api.molecular_dynamics_dir
            cfg.task_config.dir_prefix = dir_prefix
            if self.cur_iteration > 0:
                cfg.restart_point = i
            else:
                cfg.task_config.pdb_file = filename

            # Write MD yaml to tmp directory to be picked up and moved by MD job
            cfg_path = self.api.tmp_dir.joinpath(f"{dir_prefix}.yaml")
            cfg.task_config.dump_yaml(cfg_path)
            task = generate_task(cfg)
            task.arguments += ["-c", cfg_path]
            stage.add_tasks(task)

        return stage

    def generate_aggregating_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGGREGATION_STAGE_NAME
        cfg = self.cfg.aggregation_stage

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = self.api.aggregation_path(self.cur_iteration)

        # Write yaml configuration
        cfg_path = self.api.aggregation_config_path(self.cur_iteration)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_machine_learning_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MACHINE_LEARNING_STAGE_NAME
        cfg = self.cfg.machine_learning_stage

        self.api.machine_learning_path(self.cur_iteration).mkdir()

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = self.api.machine_learning_path(self.cur_iteration)
        if self.cur_iteration > 0:
            # Machine learning should use model selection API
            cfg.task_config.init_weights_path = None

        # Write yaml configuration
        cfg_path = self.api.machine_learning_config_path(self.cur_iteration)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_model_selection_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MODEL_SELECTION_STAGE_NAME
        cfg = self.cfg.model_selection_stage

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.node_local_path = self.cfg.node_local_path

        # Write yaml configuration
        cfg_path = self.api.model_selection_config_path(self.cur_iteration)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGENT_STAGE_NAME
        cfg = self.cfg.agent_stage

        self.api.agent_path(self.cur_iteration).mkdir()

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = self.api.agent_path(self.cur_iteration)

        # Write yaml configuration
        cfg_path = self.api.agent_config_path(self.cur_iteration)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
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
    num_nodes = max(1, cfg.molecular_dynamics_stage.num_jobs // cfg.gpus_per_node)

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
    pipelines = pipeline_manager.generate_pipelines()

    # Assign the workflow as a list of Pipelines to the Application Manager.
    # All the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
