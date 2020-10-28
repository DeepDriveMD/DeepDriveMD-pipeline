import os
import sys
import json
import uuid
from itertools import cycle
from pathlib import Path
from typing import List

import radical.utils as ru
from radical.entk import Pipeline, Stage, Task, AppManager


from deepdrivemd.config import ExperimentConfig, MDConfig, AggregationConfig


def get_outlier_pdbs(outlier_filename: Path) -> List[Path]:
    with open(outlier_filename) as f:
        return list(map(Path, json.load(f)))


def get_initial_pdbs(initial_pdb_dir: Path) -> List[Path]:
    """Scan input directory for PDBs and optional topologies."""

    pdb_filenames = list(initial_pdb_dir.glob("*/*.pdb"))

    if any("__" in filename.as_posix() for filename in pdb_filenames):
        raise ValueError("Initial PDB files cannot contain a double underscore __")

    return pdb_filenames


def generate_interfacing_stage(cfg: ExperimentConfig) -> Stage:
    s4 = Stage()
    s4.name = "scanning"

    # Scaning for outliers and prepare the next stage of MDs
    t4 = Task()

    t4.pre_exec = [". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh"]
    t4.pre_exec += ["conda activate %s" % cfg["conda_pytorch"]]
    t4.pre_exec += ["mkdir -p %s/Outlier_search/outlier_pdbs" % cfg["base_path"]]
    t4.pre_exec += [
        'export models=""; for i in `ls -d %s/CVAE_exps/model-cvae_runs*/`; do if [ "$models" != "" ]; then    models=$models","$i; else models=$i; fi; done;cat /dev/null'
        % cfg["base_path"]
    ]
    t4.pre_exec += ["export LANG=en_US.utf-8", "export LC_ALL=en_US.utf-8"]
    t4.pre_exec += ["unset CUDA_VISIBLE_DEVICES", "export OMP_NUM_THREADS=4"]

    cmd_cat = "cat /dev/null"
    cmd_jsrun = "jsrun -n %s -a 6 -g 6 -r 1 -c 7" % cfg["node_counts"]

    # molecules_path = '/gpfs/alpine/world-shared/ven201/tkurth/molecules/'
    t4.executable = [
        " %s; %s %s/examples/outlier_detection/run_optics_dist_summit_entk.sh"
        % (cmd_cat, cmd_jsrun, cfg["molecules_path"])
    ]
    t4.arguments = ["%s/bin/python" % cfg["conda_pytorch"]]
    t4.arguments += [
        "%s/examples/outlier_detection/optics.py" % cfg["molecules_path"],
        "--sim_path",
        "%s/MD_exps/%s" % (cfg["base_path"], cfg["system_name"]),
        "--pdb_out_path",
        "%s/Outlier_search/outlier_pdbs" % cfg["base_path"],
        "--restart_points_path",
        "%s/Outlier_search/restart_points.json" % cfg["base_path"],
        "--data_path",
        "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
        "--model_paths",
        "$models",
        "--model_type",
        cfg["model_type"],
        "--min_samples",
        10,
        "--n_outliers",
        cfg["md_counts"] + 1,
        "--dim1",
        cfg["residues"],
        "--dim2",
        cfg["residues"],
        "--cm_format",
        "sparse-concat",
        "--batch_size",
        cfg["batch_size"],
        "--distributed",
    ]

    t4.cpu_reqs = {
        "processes": 1,
        "process_type": None,
        "threads_per_process": 12,
        "thread_type": "OpenMP",
    }
    t4.gpu_reqs = {
        "processes": 1,
        "process_type": None,
        "threads_per_process": 1,
        "thread_type": "CUDA",
    }

    s4.add_tasks(t4)
    return s4


class PipelineManager:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cur_iteration = 0

        self.pipeline = Pipeline()
        pipeline.name = "DeepDriveMD"

    def _init_experiment_dir(self):
        # Name experiment directories
        self.experiment_dirs = {
            dir_name: self.cfg.experiment_directory.joinpath(dir_name)
            for dir_name in [
                "md_runs",
                "ml_runs",
                "aggregation_runs",
                "agent_runs",
                "tmp",
            ]
        }

        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        for dir_path in self.experiment_dirs.values():
            dir_path.mkdir()

    @property
    def aggregated_data_path(self):
        # Used my aggregation and ml stage
        return self.experiment_dirs["aggregation_runs"].joinpath(
            f"data_{self.cur_iteration}.h5"
        )

    def func_condition(self):
        if self.cur_iteration < self.cfg.max_iteration:
            self.func_on_true()
        else:
            self.func_on_false()

    def func_on_true(self):
        print("finishing stage %d of %d" % (self.cur_iteration, self.cfg.max_iteration))
        self._generate_pipeline_iteration()

    def func_on_false(self):
        print("Done")

    def _generate_pipeline_iteration(self):

        self.pipeline.add_stages(self.generate_md_stage())

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
        """
        Function to generate MD stage.
        """
        stage = Stage()
        stage.name = "MD"
        cfg = self.cfg.md_stage

        # TODO: factor this variable out into the config
        outlier_filename = Path("/Outlier_search/restart_points.json")

        if outlier_filename.exists():
            pdb_filenames = get_outlier_pdbs(outlier_filename)
        else:
            pdb_filenames = get_initial_pdbs(cfg.initial_pdb_dir)

        for i, pdb_filename in zip(range(cfg.num_jobs), cycle(pdb_filenames)):
            task = Task()
            task.cpu_reqs = cfg.cpu_reqs.dict()
            task.gpu_reqs = cfg.gpu_reqs.dict()
            task.pre_exec = cfg.pre_exec
            task.executable = cfg.executable
            task.arguments = cfg.arguments

            omm_dir_prefix = f"run_{self.cur_iteration:03d}_{i:04d}"

            run_config = MDConfig(
                pdb_file=pdb_filename,
                initial_pdb_dir=cfg.initial_pdb_dir,
                reference_pdb_file=cfg.reference_pdb_file,
                solvent_type=cfg.solvent_type,
                temperature_kelvin=cfg.temperature_kelvin,
                simulation_length_ns=cfg.simulation_length_ns,
                report_interval_ps=cfg.report_interval_ps,
                omm_dir_prefix=omm_dir_prefix,  # like "run_002_055",
                local_run_dir=cfg.local_run_dir,
                result_dir=self.experiment_dirs["md_runs"],
                wrap=cfg.wrap,
            )

            # Write MD yaml to tmp directory to be picked up and moved by MD job
            cfg_path = self.experiment_dirs["tmp"].joinpath(f"md-{uuid.uuid4()}.yaml")
            run_config.dump_yaml(cfg_path)

            task.arguments += ["-c", cfg_path]
            stage.add_tasks(task)

        return stage

    def generate_aggregating_stage(self) -> Stage:
        """
        Function to concatenate the MD trajectory (h5 contact map)
        """
        stage = Stage()
        stage.name = "aggregating"
        cfg = self.cfg.aggregation_stage

        # Aggregation task
        task = Task()

        task.cpu_reqs = cfg.cpu_reqs.dict()
        task.pre_exec = cfg.pre_exec
        task.executable = cfg.executable
        task.arguments = cfg.arguments

        run_config = AggregationConfig(
            rmsd=cfg.rmsd,
            fnc=cfg.fnc,
            contact_map=cfg.contact_map,
            point_cloud=cfg.point_cloud,
            last_n_h5_files=cfg.last_n_h5_files,
            verbose=cfg.verbose,
            experiment_directory=self.cfg.experiment_directory,
            out_path=self.aggregated_data_path,
        )

        # Write MD yaml to tmp directory to be picked up and moved by MD job
        cfg_path = self.experiment_dirs["aggregation_runs"].joinpath(
            f"aggregation_{self.cur_iteration}.yaml"
        )
        run_config.dump_yaml(cfg_path)

        task.arguments += ["-c", cfg_path]
        stage.add_tasks(task)

        return stage

    def generate_ml_stage(self) -> Stage:
        """
        Function to generate the learning stage
        """
        stage = Stage()
        stage.name = "learning"
        cfg = self.cfg.ml_stage

        num_ML = 1
        for i in range(num_ML):

            task = Task()
            task.cpu_reqs = cfg.cpu_reqs.dict()
            task.gpu_reqs = cfg.gpu_reqs.dict()
            task.pre_exec = cfg.pre_exec
            task.executable = cfg.executable

            cvae_dir = "fixme"
            hp = cfg["ml_hpo"][i]
            task.arguments = [
                "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
                "./",
                cvae_dir,
                str(cfg["residues"]),
                str(hp["latent_dim"]),
                "non-amp",
                "distributed",
                str(hp["batch_size"]),
                str(cfg["epoch"]),
                str(cfg["sample_interval"]),
                hp["optimizer"],
                hp["loss_weights"],
                cfg["init_weights"],
            ]

            stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        return Stage()


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
    num_nodes = max(1, cfg.md_stage.num_jobs // cfg.gpus_per_node)

    res_dict = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus_per_node * cfg.hardware_threads_per_cpu * num_nodes,
        "gpus": cfg.gpus_per_node * num_nodes,
    }

    appman.resource_desc = res_dict

    pipeline_manager = PipelineManager(cfg)
    pipeline = pipeline_manager.generate_pipeline()

    pipelines = [pipeline]

    # Assign the workflow as a list of Pipelines to the Application Manager. In
    # this way, all the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
