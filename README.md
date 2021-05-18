# DeepDriveMD-F (DeepDriveMD-pipeline)

DeepDriveMD-F: Deep-Learning Driven Adaptive Molecular Simulations (file-based continual learning loop)

[![Documentation Status](https://readthedocs.org/projects/deepdrivemd-pipeline/badge/?version=latest)](https://deepdrivemd-pipeline.readthedocs.io/en/latest/?badge=latest)

Details can be found in the [ducumentation](https://deepdrivemd-pipeline.readthedocs.io/en/latest/). For more information, please see our [website](https://deepdrivemd.github.io/).

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

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
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

3. Install the `h5py` package version 2.10.0:

```
conda install h5py=2.10.0
```

4. Install the `tensorflow-gpu` package (need to compile with CUDA 11.1.1, not compatible with CUDA 10.1.243):

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
