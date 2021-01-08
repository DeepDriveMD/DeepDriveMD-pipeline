# DeepDriveMD-pipeline

DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations

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
Where the topology files are optional and only used when `md_stage.solvent_type` is "explicit". Only one system directory is needed but an arbitrary number is supported. Also note that the system directory names are arbitrary. The path to the `data` directory should be passed into the config via `md_stage.initial_configs_dir`.
