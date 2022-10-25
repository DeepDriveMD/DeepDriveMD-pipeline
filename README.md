# DeepDriveMD-F (DeepDriveMD-pipeline)

DeepDriveMD-F: Deep-Learning Driven Adaptive Molecular Simulations (file-based continual learning loop)

[![Documentation Status](https://readthedocs.org/projects/deepdrivemd-pipeline/badge/?version=latest)](https://deepdrivemd-pipeline.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [documentation](https://deepdrivemd-pipeline.readthedocs.io/en/latest/). For more information, please see our [website](https://deepdrivemd.github.io/).

## How to run

Running DeepDriveMD requires the use of virtual environment. At this point we distinguish different stage runs of DeepDriveMD using different virtual environments to alleviate package compatibility with associated dependencies across different stages.

For instance, below is a list of Python versions used by different virtual environments:

- RCT env: Python 3.7.8
- OpenMM env: Python 3.7.9
- pytorch (AAE) env: Python 3.7.9
- keras-cvae (CVAE) & rapids-dbscan: Python 3.6.12

### Setup

#### Stage: molecular_dynamics

1. Install `deepdrivemd` into a virtualenv with a Python virtual environment:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

Or with a Conda virtual environment:

```
. ~/miniconda3/etc/profile.d/conda.sh
conda create -n deepdrivemd python=3.7.9
conda activate deepdrivemd
pip install --upgrade pip setuptools wheel
conda install scipy (this step is needed if a failure of installing scipy is observed)
pip install -e .
```

2. Install OpenMM:

- by source code (for Linux ppc64le, e.g., Summit)
https://gist.github.com/lee212/4bbfe520c8003fbb91929731b8ea8a1e

- by conda (for Linux x86\_64, e.g., PSC Bridges)
```
module load anaconda3
module load cuda/9.2
source /opt/packages/anaconda/anaconda3-5.2.0/etc/profile.d/conda.sh
conda install -c omnia/label/cuda92 openmm
```

3. In some places, DeepDriveMD relies on external libraries to configure MD simulations and import specific ML models.

For MD, install the `mdtools` package found here: https://github.com/braceal/MD-tools

```
git clone https://github.com/braceal/MD-tools.git
pip install .
```

For ML (specifically the AAE model), install the `molecules` package found here: https://github.com/braceal/molecules/tree/main

```
git clone https://github.com/braceal/molecules.git
pip install .
```

#### Stage: machine_learning

1. Install the `deepdrivemd` virtual environment as above (`deepdrivemd` is needed in all the virtual environments since each task uses the DDMD_API to communicate with the outputs of other tasks).

2. Install the `keras-CVAE` model with `rapidsai DBSCAN` package found here: https://www.ibm.com/docs/en/wmlce/1.6.2?topic=installing-mldl-frameworks

```
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda install powerai-rapids
```

3. Install packages `scikit-learn` and `h5py` version 2.10.0:

```
conda install scikit-learn h5py=2.10.0
```

4. Install the `tensorflow-gpu` package (need to compile with CUDA 10.2.89, not compatible with CUDA 10.1.243 and CUDA 11.1.1 or higher versions):

```
conda install tensorflow-gpu
```

### Generating a YAML input spec:

First, run this command to get a _sample_ YAML config file:

```
python -m deepdrivemd.config
```

This will write a file named `deepdrivemd_template.yaml` which should be adapted for the experiment at hand. You should configure the `molecular_dynamics_stage`, `aggregation_stage`, `machine_learning_stage`, `model_selection_stage` and `agent_stage` sections to use the appropriate run commands and environment setups.

### Running an experiment

Then, launch an experiment with:

```
python -m deepdrivemd.deepdrivemd -c <experiment_config.yaml>
```

This experiment should be launched

### Note on input data

The input PDB and topology files should have the following structure:

```
ls data/sys*

data/sys1:
comp.pdb comp.top

data/sys2:
comp.pdb comp.top
```
Where the topology files are optional and only used when `molecular_dynamics_stage.task_config.solvent_type` is "explicit". Only one system directory is needed but an arbitrary number are supported. Also note that the system directory names are arbitrary. The path to the `data` directory should be passed into the config via `molecular_dynamics_stage.initial_pdb_dir`.

# DeepDriveMD-S (Streaming asynchronous execution with ADIOS)

The streaming version of DeepDriveMD uses the adios2 package.

`adios2` is installed with spack:
```
spack install adios2 +python -mpi
```

To use adios2 in python, one needs to load the corresponding module, for example, with
```
module load adios2
```
or
```
spack load adios2
```
and to set up `PYTHONPATH` to the corresponding subdirectory of the adios2 installation: 
```
export PYTHONPATH=<ADIOS2_dir>/lib/python<version>/site-packages/:$PYTHONPATH
```

To make a small 30m, 12 simulation, 1 aggregator, test run of DeepDriveMD-S, cd into `test/` and run
```
make run1
```
To make a large 12h, 120 simulations, 10 aggregators run do
```
make run2
```
in DeepDriveMD-pipeline directory.

To watch how one of the aggregation files grows, do, for example
```
make watch1 d=3	
```
assuming that the experiment directory is `../Outputs/3`.

To watch what happens in one of the simulation task directory, do
```
make watch2 d=3
```

To watch the log for task 0014 (for run1 it corresponds to the outlier search log), do
```
make watch3 d=0014
```

To clean after the run, do
```
make clean d=3
```


The configuration files for the run, including `generate.py` that is used to create `config.yaml`, adios xml files for SST streams between simulations and aggregators and for BP files between aggregators and the downstream two components, are in a subdirectory of
test/bba, for example, `test1_stream` (run1) and `lassen-keras-dbscan_stream` (run2). Yaml files are generated by running `./generate.py > config.yaml` or, if you prefer, you can edit `config.yaml` directly and not use `generate.py`.

To use multiple input files, put the corresponding pdb files into `cfg.initial_pdb_dir`. The simulation sorts pdb files from this directory and picks up the one corresponding to its task id modulo the number of pdb files.

# Contributing

Please report **bugs**, **feature requests**, or **questions** through the [Issue Tracker](https://github.com/DeepDriveMD/DeepDriveMD-pipeline/issues).

If you are looking to contribute, please see [`CONTRIBUTING.md`](https://github.com/DeepDriveMD/DeepDriveMD-pipeline/blob/main/CONTRIBUTING.md).

# License

DeepDriveMD has a MIT license, as seen in the [LICENSE](https://github.com/DeepDriveMD/DeepDriveMD-pipeline/blob/main/LICENSE.md) file.
