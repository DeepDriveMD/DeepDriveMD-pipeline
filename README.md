# DeepDriveMD-pipeline

DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations

## How to run

### Setup

Install `deepdrivemd` into a virtualenv with:

```
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

This will write a file named `deepdrivemd_template.yaml` which should be adapted for the experiment at hand. You should configure the `md_stage`, `aggregation_stage`, `ml_stage` and `od_stage` sections to use the appropriate run commands and environment setups.

### Running an experiment

Then, launch an experiment with:

```
python -m deepdrivemd.deepdrivemd <experiment_config.yaml>
```

This experiment should be launched