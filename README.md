# DeepDriveMD-F (DeepDriveMD-pipeline)

DeepDriveMD-F: Deep-Learning Driven Adaptive Molecular Simulations (file-based continual learning loop)

[![Documentation Status](https://readthedocs.org/projects/deepdrivemd-pipeline/badge/?version=latest)](https://deepdrivemd-pipeline.readthedocs.io/en/latest/?badge=latest)

## How to run

### Setup

Install `deepdrivemd` into a virtualenv with:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

```
pre-commit install
pre-commit autoupdate
```

In some places, DeepDriveMD relies on external libraries to configure MD simulations and import specific ML models.

For MD, install the `mdtools` package found here: https://github.com/braceal/MD-tools

For ML (specifically the AAE model), install the `molecules` package found here: https://github.com/braceal/molecules/tree/main

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
