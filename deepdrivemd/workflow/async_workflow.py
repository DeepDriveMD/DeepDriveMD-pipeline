import itertools
from typing import List
from radical.entk import Pipeline, Stage
from deepdrivemd.config import ExperimentConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.workflow.utils import generate_task


class AsyncPipelineManager:

    MOLECULAR_DYNAMICS_PIPELINE_NAME = "MolecularDynamics"
    AGGREGATION_PIPELINE_NAME = "Aggregating"
    MACHINE_LEARNING_PIPELINE_NAME = "MachineLearning"
    MODEL_SELECTION_PIPELINE_NAME = "ModelSelection"
    AGENT_PIPELINE_NAME = "Agent"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.stage_idx = 0  # This is never changed
        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_stage.runs_dir.mkdir()
        self.api.aggregation_stage.runs_dir.mkdir()
        self.api.machine_learning_stage.runs_dir.mkdir()
        self.api.model_selection_stage.runs_dir.mkdir()
        self.api.agent_stage.runs_dir.mkdir()

    def generate_pipelines(self) -> List[Pipeline]:
        pipelines = [
            self.generate_molecular_dynamics_pipeline(),
            self.generate_aggregating_pipeline(),
            self.generate_machine_learning_pipeline(),
            self.generate_model_selection_pipeline(),
            self.generate_agent_pipeline(),
        ]
        return pipelines

    def generate_molecular_dynamics_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = self.MOLECULAR_DYNAMICS_PIPELINE_NAME
        cfg = self.cfg.molecular_dynamics_stage
        stage_api = self.api.molecular_dynamics_stage

        filenames = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
        filenames = itertools.cycle(filenames)

        for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path

            cfg.task_config.pdb_file = next(filenames)

            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            cfg.task_config.dump_yaml(cfg_path)
            task = generate_task(cfg)
            task.arguments += ["-c", cfg_path.as_posix()]
            stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = self.MOLECULAR_DYNAMICS_PIPELINE_NAME
        pipeline.add_stages(stage)

        return pipeline

    def generate_aggregating_pipeline(self) -> Pipeline:
        # TODO: add support for multiple aggregator tasks
        stage = Stage()
        stage.name = self.AGGREGATION_PIPELINE_NAME
        cfg = self.cfg.aggregation_stage
        stage_api = self.api.aggregation_stage

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

        pipeline = Pipeline()
        pipeline.name = self.AGGREGATION_PIPELINE_NAME
        pipeline.add_stages(stage)

        return pipeline

    def generate_machine_learning_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = self.MACHINE_LEARNING_PIPELINE_NAME
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

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = self.MACHINE_LEARNING_PIPELINE_NAME
        pipeline.add_stages(stage)

        return pipeline

    def generate_model_selection_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = self.MODEL_SELECTION_PIPELINE_NAME
        cfg = self.cfg.model_selection_stage
        stage_api = self.api.model_selection_stage

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

        pipeline = Pipeline()
        pipeline.name = self.MODEL_SELECTION_PIPELINE_NAME
        pipeline.add_stages(stage)

        return pipeline

    def generate_agent_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = self.AGENT_PIPELINE_NAME
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

        pipeline = Pipeline()
        pipeline.name = self.AGENT_PIPELINE_NAME
        pipeline.add_stages(stage)

        return pipeline
