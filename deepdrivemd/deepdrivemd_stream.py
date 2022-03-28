import os
import shutil
from typing import List
import radical.utils as ru
from radical.entk import AppManager, Pipeline, Stage, Task
from deepdrivemd.config import StreamingExperimentConfig, BaseStageConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.utils import parse_args
import math
from pathlib import Path


def generate_task(cfg: BaseStageConfig) -> Task:
    task = Task()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    task.pre_exec = cfg.pre_exec.copy()
    task.executable = cfg.executable
    task.arguments = cfg.arguments.copy()
    return task


class PipelineManager:
    MOLECULAR_DYNAMICS_STAGE_NAME = "MolecularDynamics"
    AGGREGATION_STAGE_NAME = "Aggregating"
    MACHINE_LEARNING_STAGE_NAME = "MachineLearning"
    AGENT_STAGE_NAME = "Agent"

    MOLECULAR_DYNAMICS_PIPELINE_NAME = "MolecularDynamicsPipeline"
    AGGREGATION_PIPELINE_NAME = "AggregatingPipeline"
    MACHINE_LEARNING_PIPELINE_NAME = "MachineLearningPipeline"
    AGENT_PIPELINE_NAME = "AgentPipeline"

    def __init__(self, cfg: StreamingExperimentConfig):
        self.cfg = cfg
        self.stage_idx = 0
        self.api = DeepDriveMD_API(cfg.experiment_directory)

        self.pipelines = {}

        p_md = Pipeline()
        p_md.name = self.MOLECULAR_DYNAMICS_PIPELINE_NAME

        self.pipelines[p_md.name] = p_md

        p_aggregate = Pipeline()
        p_aggregate.name = self.AGGREGATION_PIPELINE_NAME

        self.pipelines[p_aggregate.name] = p_aggregate

        p_ml = Pipeline()
        p_ml.name = self.MACHINE_LEARNING_PIPELINE_NAME
        self.pipelines[p_ml.name] = p_ml

        p_outliers = Pipeline()
        p_outliers.name = self.AGENT_PIPELINE_NAME
        self.pipelines[p_outliers.name] = p_outliers

        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_stage.runs_dir.mkdir()
        self.api.aggregation_stage.runs_dir.mkdir()
        self.api.machine_learning_stage.runs_dir.mkdir()
        self.api.agent_stage.runs_dir.mkdir()

    def _generate_pipeline_iteration(self):

        self.pipelines[self.MOLECULAR_DYNAMICS_PIPELINE_NAME].add_stages(
            self.generate_molecular_dynamics_stage()
        )
        self.pipelines[self.AGGREGATION_PIPELINE_NAME].add_stages(
            self.generate_aggregating_stage()
        )
        self.pipelines[self.MACHINE_LEARNING_PIPELINE_NAME].add_stages(
            self.generate_machine_learning_stage()
        )
        self.pipelines[self.AGENT_PIPELINE_NAME].add_stages(self.generate_agent_stage())

        self.stage_idx += 1

    def generate_pipelines(self) -> List[Pipeline]:
        self._generate_pipeline_iteration()
        return list(self.pipelines.values())

    def generate_molecular_dynamics_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MOLECULAR_DYNAMICS_STAGE_NAME
        cfg = self.cfg.molecular_dynamics_stage
        stage_api = self.api.molecular_dynamics_stage

        for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            cfg.task_config.dump_yaml(cfg_path)
            task = generate_task(cfg)
            task.arguments += ["-c", cfg_path.as_posix()]
            stage.add_tasks(task)

        return stage

    def generate_aggregating_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGGREGATION_STAGE_NAME
        cfg = self.cfg.aggregation_stage
        stage_api = self.api.aggregation_stage

        for task_idx in range(cfg.num_tasks):
            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            # Write yaml configuration
            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            cfg.task_config.dump_yaml(cfg_path)
            task = generate_task(cfg)
            task.arguments += ["-c", cfg_path.as_posix()]
            stage.add_tasks(task)

        return stage

    def generate_machine_learning_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MACHINE_LEARNING_STAGE_NAME
        cfg = self.cfg.machine_learning_stage
        stage_api = self.api.machine_learning_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path
        cfg.task_config.model_tag = stage_api.unique_name(output_path)
        if self.stage_idx > 0:
            # Machine learning should use model selection API
            cfg.task_config.init_weights_path = None

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGENT_STAGE_NAME
        cfg = self.cfg.agent_stage
        stage_api = self.api.agent_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage


def compute_number_of_nodes(cfg: StreamingExperimentConfig) -> int:
    nodes = 0

    for stage in (
        cfg.molecular_dynamics_stage,
        cfg.aggregation_stage,
        cfg.machine_learning_stage,
        cfg.agent_stage,
    ):
        nodes_cpu = (
            stage.cpu_reqs.processes
            * stage.cpu_reqs.threads_per_process
            * stage.num_tasks
        ) / (cfg.cpus_per_node * cfg.hardware_threads_per_cpu)
        nodes_gpu = (
            stage.gpu_reqs.processes
            * stage.gpu_reqs.threads_per_process
            * stage.num_tasks
        ) / cfg.gpus_per_node
        nodes += max(nodes_cpu, nodes_gpu)
    return int(math.ceil(nodes))


if __name__ == "__main__":

    args = parse_args()
    cfg = StreamingExperimentConfig.from_yaml(args.config)
    cfg.config_directory = os.path.dirname(os.path.abspath(args.config))
    print("config_directory = ", cfg.config_directory)
    print("experiment directory = ", cfg.experiment_directory)

    cfg.adios_xml_sim = Path(cfg.config_directory) / "adios_sim.xml"
    cfg.adios_xml_agg = Path(cfg.config_directory) / "adios_agg.xml"
    cfg.adios_xml_agg_4ml = Path(cfg.config_directory) / "adios_agg_4ml.xml"
    cfg.adios_xml_file = Path(cfg.config_directory) / "adios_file.xml"

    cfg.agent_stage.task_config.adios_xml_agg = cfg.adios_xml_agg
    cfg.aggregation_stage.task_config.adios_xml_agg = cfg.adios_xml_agg
    cfg.aggregation_stage.task_config.adios_xml_agg_4ml = cfg.adios_xml_agg_4ml
    cfg.machine_learning_stage.task_config.adios_xml_agg = cfg.adios_xml_agg
    cfg.machine_learning_stage.task_config.adios_xml_agg_4ml = cfg.adios_xml_agg_4ml
    cfg.molecular_dynamics_stage.task_config.adios_xml_sim = cfg.adios_xml_sim
    cfg.molecular_dynamics_stage.task_config.adios_xml_file = cfg.adios_xml_file

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

    num_nodes = compute_number_of_nodes(cfg)

    print(f"Required number of nodes: {num_nodes}")

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
    # Back up configuration directory
    shutil.copytree(cfg.config_directory, cfg.experiment_directory / "etc")

    pipelines = pipeline_manager.generate_pipelines()
    # Assign the workflow as a list of Pipelines to the Application Manager.
    # All the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
