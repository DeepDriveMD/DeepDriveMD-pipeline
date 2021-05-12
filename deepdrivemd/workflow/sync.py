import itertools
from typing import List
from radical.entk import Pipeline, Stage
from deepdrivemd.config import ExperimentConfig
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.workflow.utils import generate_task


class SyncPipelineManager:

    PIPELINE_NAME = "DeepDriveMD"
    MOLECULAR_DYNAMICS_STAGE_NAME = "MolecularDynamics"
    AGGREGATION_STAGE_NAME = "Aggregating"
    MACHINE_LEARNING_STAGE_NAME = "MachineLearning"
    MODEL_SELECTION_STAGE_NAME = "ModelSelection"
    AGENT_STAGE_NAME = "Agent"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.stage_idx = 0

        self.api = DeepDriveMD_API(cfg.experiment_directory)
        self.pipeline = Pipeline()
        self.pipeline.name = self.PIPELINE_NAME

        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_stage.runs_dir.mkdir()
        self.api.aggregation_stage.runs_dir.mkdir()
        self.api.machine_learning_stage.runs_dir.mkdir()
        self.api.model_selection_stage.runs_dir.mkdir()
        self.api.agent_stage.runs_dir.mkdir()

    def func_condition(self):
        if self.stage_idx < self.cfg.max_iteration:
            self.func_on_true()
        else:
            self.func_on_false()

    def func_on_true(self):
        print(f"Finishing stage {self.stage_idx} of {self.cfg.max_iteration}")
        self._generate_pipeline_iteration()

    def func_on_false(self):
        print("Done")

    def _generate_pipeline_iteration(self):

        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())

        if not self.cfg.aggregation_stage.skip_aggregation:
            self.pipeline.add_stages(self.generate_aggregating_stage())

        if self.stage_idx % self.cfg.machine_learning_stage.retrain_freq == 0:
            self.pipeline.add_stages(self.generate_machine_learning_stage())
        self.pipeline.add_stages(self.generate_model_selection_stage())

        agent_stage = self.generate_agent_stage()
        agent_stage.post_exec = self.func_condition
        self.pipeline.add_stages(agent_stage)

        self.stage_idx += 1

    def generate_pipelines(self) -> List[Pipeline]:
        self._generate_pipeline_iteration()
        return [self.pipeline]

    def generate_molecular_dynamics_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MOLECULAR_DYNAMICS_STAGE_NAME
        cfg = self.cfg.molecular_dynamics_stage
        stage_api = self.api.molecular_dynamics_stage

        if self.stage_idx == 0:
            filenames = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
            filenames = itertools.cycle(filenames)
        else:
            filenames = None

        for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path
            if self.stage_idx == 0:
                assert filenames is not None
                cfg.task_config.pdb_file = next(filenames)
            else:
                cfg.task_config.pdb_file = None

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

    def generate_model_selection_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MODEL_SELECTION_STAGE_NAME
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