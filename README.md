# DeepDriveMD-F (DeepDriveMD-pipeline)

DeepDriveMD-F: Deep-Learning Driven Adaptive Molecular Simulations (file-based continual learning loop)

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

# DeepDriveMD-S

The streaming version of DeepDriveMD uses two extra packages: adios2 and lockfile

`lockfile` is trivially installed with conda:
```
conda install lockfile
```

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

To make a small 30m, 12 simulation, 1 aggregator, test run of DeepDriveMD-S, do
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

To push the code to git:
```
make push m="comment"
```
At the moment, only a single word comment works.


