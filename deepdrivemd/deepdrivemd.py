import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import List

import radical.utils as ru
from radical.entk import Pipeline, Stage, Task, AppManager


from deepdrivemd.config import ExperimentConfig


def get_ligand_topology(pdb_file):
    # Assume abspath is passed and that ligand ID is enoded in pdb_file name
    # pdb_file: /path/system_<ligid>.pdb
    # topol:    /path/topology_<ligid>.pdb
    topol_dir = "/gpfs/alpine/med110/scratch/atrifan2/PLPro_ligands/gb_plpro/DrugWorkflows/workflow-2/top_dir/"
    pdb_filename = os.path.basename(pdb_file)
    ligid = pdb_filename.split("_")[1]
    if ".pdb" in ligid:
        ligid = ligid[:-4]
    return os.path.join(topol_dir, f"topology_{ligid}.top")


def get_outliers(outlier_filename: Path) -> List[str]:
    if not os.path.exists(outlier_filename):
        return []
    with open(outlier_filename) as f:
        return json.load(f)


def generate_MD_stage(num_MD=1):
    """
    Function to generate MD stage.
    """
    s1 = Stage()
    s1.name = "MD"

    # TODO: factor this variable out into the config
    outlier_filename = Path("/Outlier_search/restart_points.json")
    outliers = get_outliers(outlier_filename)

    # MD tasks
    global time_stamp
    time_stamp = int(time.time())

    for i in range(num_MD):
        t1 = Task()

        # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_exps/fs-pep/run_openmm.py
        t1.pre_exec = cfg.md_runner.pre_exec

        t1.executable = ["%s/bin/python" % cfg["conda_openmm"]]  # run_openmm.py
        t1.arguments = [
            "%s/MD_exps/%s/run_openmm.py" % (cfg["base_path"], cfg["system_name"])
        ]
        # t1.arguments += ['--topol', '%s/MD_exps/fs-pep/pdb/topol.top' % cfg['base_path']]

        # if 'top_file' in cfg:
        #    t1.arguments += ['--topol', cfg['top_file']]

        # pick initial point of simulation
        if not outliers:
            # Not used ince outliers is filled from the start
            t1.arguments += ["--pdb_file", cfg["pdb_file"]]
            t1.arguments += ["--topol", get_ligand_topology(cfg["pdb_file"])]
            print("stage 1 error. using pdb in config")
        elif outliers[i].endswith("pdb"):
            print("Getting PDB outlier")
            print("pdb:", outliers[i])
            print("top:", get_ligand_topology(outliers[i]))
            t1.arguments += [
                "--pdb_file",
                outliers[i],
                "--topol",
                get_ligand_topology(outliers[i]),
            ]
            t1.pre_exec += ["cp %s ./" % outliers[i]]

        # how long to run the simulation
        t1.arguments += ["--length", cfg.md_runner.simulation_length_ns]

        # assign hardware the task
        t1.cpu_reqs = {
            "processes": 1,
            "process_type": None,
            "threads_per_process": 4,
            "thread_type": "OpenMP",
        }
        t1.gpu_reqs = {
            "processes": 1,
            "process_type": None,
            "threads_per_process": 1,
            "thread_type": "CUDA",
        }

        # Add the MD task to the simulating stage
        s1.add_tasks(t1)

    return s1


def generate_preprocessing_stage(num_MD=1):

    global time_stamp
    s_1 = Stage()
    s_1.name = "preprocessing"

    for i in range(num_MD):

        t_1 = Task()

        omm_dir = "omm_runs_%s" % (time_stamp + i)

        print("omm_dir", omm_dir)

        t_1.pre_exec = [
            ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
            "conda activate %s" % cfg["conda_pytorch"],
            "export LANG=en_US.utf-8",
            "export LC_ALL=en_US.utf-8",
            "cd %s/MD_exps/%s/%s/" % (cfg["base_path"], cfg["system_name"], omm_dir),
            "export PDB_FILE=$(ls | grep .pdb)",
        ]

        t_1.executable = ["%s/bin/python" % (cfg["conda_pytorch"])]

        output_h5 = f'{cfg["h5_tmp_dir"]}/output_{uuid.uuid4()}.h5'

        t_1.arguments = [
            "%s/scripts/traj_to_dset.py" % cfg["molecules_path"],
            "-t",
            ".",  # 'output.dcd',
            "-p",
            "${PDB_FILE}",
            "-r",
            "${PDB_FILE}",
            "-o",
            "%s" % output_h5,
            "--contact_maps_parameters",
            "kernel_type=threshold,threshold=%s" % cfg["cutoff"],
            "-s",
            cfg["selection"],
            "--rmsd",
            "--fnc",
            "--contact_map",
            "--point_cloud",
            "--verbose",
        ]

        # Add the aggregation task to the aggreagating stage
        t_1.cpu_reqs = {
            "processes": 1,
            "process_type": None,
            "threads_per_process": 26,
            "thread_type": "OpenMP",
        }

        s_1.add_tasks(t_1)

    return s_1


def generate_aggregating_stage():
    """
    Function to concatenate the MD trajectory (h5 contact map)
    """
    s2 = Stage()
    s2.name = "aggregating"

    # Aggregation task
    t2 = Task()

    # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/MD_to_CVAE/MD_to_CVAE.py
    t2.pre_exec = [
        ". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh",
        "conda activate %s" % cfg["conda_pytorch"],
        "export LANG=en_US.utf-8",
        "export LC_ALL=en_US.utf-8",
    ]
    # preprocessing for molecules' script, it needs files in a single
    # directory
    # the following pre-processing does:
    # 1) find all (.dcd) files from openmm results
    # 2) create a temp directory
    # 3) symlink them in the temp directory
    #

    # t2.pre_exec += [
    #        'export dcd_list=(`ls %s/MD_exps/%s/omm_runs_*/*dcd`)' % (cfg['base_path'], cfg['system_name']),
    #        'export tmp_path=`mktemp -p %s/MD_to_CVAE/ -d`' % cfg['base_path'],
    #        'for dcd in ${dcd_list[@]}; do tmp=$(basename $(dirname $dcd)); ln -s $dcd $tmp_path/$tmp.dcd; done',
    #        'ln -s %s $tmp_path/prot.pdb' % cfg['pdb_file'],
    #        'ls ${tmp_path}']

    # t2.pre_exec += ['unset CUDA_VISIBLE_DEVICES', 'export OMP_NUM_THREADS=4']

    # node_cnt_constraint = cfg['md_counts'] * max(1, CUR_STAGE) // 12
    # cmd_cat    = 'cat /dev/null'
    # cmd_jsrun  = 'jsrun -n %s -r 1 -a 6 -c 7 -d packed' % min(cfg['node_counts'], node_cnt_constraint)

    t2.executable = ["%s/bin/python" % cfg["conda_pytorch"]]  # MD_to_CVAE.py

    t2.arguments = [
        "%s/scripts/concat_dsets.py" % cfg["molecules_path"],
        "-d",
        cfg["h5_tmp_dir"],
        "-o",
        "%s/MD_to_CVAE/cvae_input.h5" % cfg["base_path"],
        "--rmsd",
        "--fnc",
        "--contact_map",
        "--point_cloud",
        "--verbose",
    ]

    # Add the aggregation task to the aggreagating stage
    t2.cpu_reqs = {
        "processes": 1,
        "process_type": None,
        "threads_per_process": 26,
        "thread_type": "OpenMP",
    }

    s2.add_tasks(t2)
    return s2


def generate_ML_stage(num_ML=1):
    """
    Function to generate the learning stage
    """
    # learn task
    time_stamp = int(time.time())
    stages = []
    for i in range(num_ML):
        s3 = Stage()
        s3.name = "learning"

        t3 = Task()
        # https://github.com/radical-collaboration/hyperspace/blob/MD/microscope/experiments/CVAE_exps/train_cvae.py
        t3.pre_exec = [". /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh"]
        t3.pre_exec += [
            "module load gcc/7.4.0",
            "module load cuda/10.1.243",
            "module load hdf5/1.10.4",
            "export LANG=en_US.utf-8",
            "export LC_ALL=en_US.utf-8",
        ]
        t3.pre_exec += ["conda activate %s" % cfg["conda_pytorch"]]
        dim = i + 3
        cvae_dir = "cvae_runs_%.2d_%d" % (dim, time_stamp + i)
        t3.pre_exec += ["cd %s/CVAE_exps" % cfg["base_path"]]
        t3.pre_exec += [
            "export LD_LIBRARY_PATH=/gpfs/alpine/proj-shared/med110/atrifan/scripts/cuda/targets/ppc64le-linux/lib/:$LD_LIBRARY_PATH"
        ]
        # t3.pre_exec += ['mkdir -p %s && cd %s' % (cvae_dir, cvae_dir)] # model_id creates sub-dir
        # this is for ddp, distributed
        t3.pre_exec += ["unset CUDA_VISIBLE_DEVICES", "export OMP_NUM_THREADS=4"]
        # pnodes = cfg['node_counts'] // num_ML # partition
        pnodes = 1  # max(1, pnodes)

        hp = cfg["ml_hpo"][i]
        cmd_cat = "cat /dev/null"
        cmd_jsrun = "jsrun -n %s -r 1 -g 6 -a 6 -c 42 -d packed" % pnodes

        # VAE config
        # cmd_vae    = '%s/examples/run_vae_dist_summit_entk.sh' % cfg['molecules_path']
        # cmd_sparse = ' '.join(['%s/MD_to_CVAE/cvae_input.h5' % cfg["base_path"],
        #                        "./", cvae_dir, 'sparse-concat', 'resnet',
        #                        str(cfg['residues']), str(cfg['residues']),
        #                        str(hp['latent_dim']), 'amp', 'non-distributed',
        #                        str(hp['batch_size']), str(cfg['epoch']),
        #                        str(cfg['sample_interval']),
        #                        hp['optimizer'], cfg['init_weights']])

        # AAE config
        cmd_vae = (
            "%s/examples/bin/summit/run_aae_dist_summit_entk.sh" % cfg["molecules_path"]
        )
        t3.executable = ["%s; %s %s" % (cmd_cat, cmd_jsrun, cmd_vae)]
        t3.arguments = [
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

        # + f'{cfg['molecules_path']}/examples/run_vae_dist_summit.sh -i {sparse_matrix_path} -o ./ --model_id {cvae_dir} -f sparse-concat -t resnet --dim1 168 --dim2 168 -d 21 --amp --distributed -b {batch_size} -e {epoch} -S 3']
        #     ,
        #             '-i', sparse_matrix_path,
        #             '-o', './',
        #             '--model_id', cvae_dir,
        #             '-f', 'sparse-concat',
        #             '-t', 'resnet',
        #             # fs-pep
        #             '--dim1', 168,
        #             '--dim2', 168,
        #             '-d', 21,
        #             '--amp',      # sparse matrix
        #             '--distributed',
        #             '-b', batch_size, # batch size
        #             '-e', epoch,# epoch
        #             '-S', 3
        #             ]

        t3.cpu_reqs = {
            "processes": 1,
            "process_type": "MPI",
            "threads_per_process": 4,
            "thread_type": "OpenMP",
        }
        t3.gpu_reqs = {
            "processes": 1,
            "process_type": None,
            "threads_per_process": 1,
            "thread_type": "CUDA",
        }

        # Add the learn task to the learning stage
        s3.add_tasks(t3)
        stages.append(s3)
    return stages


def generate_interfacing_stage():
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
    def __init__(self, cfg: ExperimentConfig, pipeline_name: str):
        self.cfg = cfg
        self.cur_iteration = 0

        self.pipeline = Pipeline()
        pipeline.name = pipeline_name

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

        s1 = generate_MD_stage(cfg)
        # Add simulating stage to the training pipeline
        self.pipeline.add_stages(s1)

        # Data Preprocessing stage
        s_1 = generate_preprocessing_stage(cfg)
        self.pipeline.add_stages(s_1)

        # --------------------------
        # Aggregate stage
        s2 = generate_aggregating_stage(cfg)
        self.pipeline.add_stages(s2)

        if self.cur_iteration % cfg["RETRAIN_FREQ"] == 0:
            # --------------------------
            # Learning stage
            s3 = generate_ML_stage(cfg)
            # Add the learning stage to the pipeline
            self.pipeline.add_stages(s3)

        # --------------------------
        # Outlier identification stage
        s4 = generate_interfacing_stage(cfg)
        s4.post_exec = self.func_condition
        self.pipeline.add_stages(s4)

        self.cur_iteration += 1

    def generate_pipeline(self):
        self._generate_pipeline_iteration()
        return self.pipeline


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
    num_nodes = max(1, cfg.md_runner.num_jobs // cfg.gpus_per_node)

    res_dict = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": 42 * 4 * num_nodes,
        "gpus": 6 * num_nodes,
    }

    appman.resource_desc = res_dict

    pipeline_manager = PipelineManager(cfg, "DeepDriveMD")
    pipeline = pipeline_manager.generate_pipeline()

    pipelines = [pipeline]

    # Assign the workflow as a list of Pipelines to the Application Manager. In
    # this way, all the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
