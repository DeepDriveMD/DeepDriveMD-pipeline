# Software package
DeepDriveMD consists of multiple stages including MD simulation, ML training
and each stage require specific software such as OpenMM CUDA, Pytorch, MD-tools
and molecules. Luckily these requirement can be installed via pip or conda.

```
module load cuda/10.2.0
conda install -c omnia-dev/label/cuda102 openmm
pip install git+https://github.com/braceal/MD-tools.git
pip install git+https://github.com/braceal/molecules.git
```

Due to the conflict of dependency libraries, we need to create separate
environment for pytorch, for example in conda:

```
conda create -n pytorch python=3.6.12
conda activate pytorch
conda install pytorch
```

On PSC Bridges2, there is a sample conda env for openMM 7.5.1,:

```
conda activate /ocean/projects/mcb110096p/hrlee/conda/1.6.6
```

and for pytorch:

```
conda activate /ocean/projects/mcb110096p/hrlee/conda/pytorch
```

# Example template

DeepDriveMD runs with *user-defined* configuration file, which can be created
by an empty template, for example:

```
python -m deepdrivemd.config
```

this command calls deepdrivemd.config module, and generates
`deepdrvemd_template.yaml`. Users can start modifying values in the template
such as HPC resource name, experiment base directory and input pdb path.

There is a sample template for PSC Bridges:

```
/ocean/projects/mcb110096p/hrlee/git/DeepDriveMD-pipeline/experiment/deepdrivemd_bridges.yaml
```

**NOTE** that `experiment_directory` has to be non-exist prior actual run, it
is a new base directory to be created. Replace the current value with your home
location:

```
experiment_directory: /ocean/projects/mcb110096p/hrlee/git/DeepDriveMD-pipeline/experiment/1st_run
```

# Test Run

```
$ python -m deepdrivemd.deepdrivemd -c deepdrivemd_bridges.yaml
```

The example output looks like:

```
================================================================================
 COVID-19 - Workflow2
================================================================================

EnTK session: re.session.br013.ib.bridges2.psc.edu.hrlee.018784.0008
Creating AppManagerSetting up RabbitMQ system                                 ok
                                                                              ok
Validating and assigning resource manager                                     ok
Setting up RabbitMQ system                                                   n/a
new session: [re.session.br013.ib.bridges2.psc.edu.hrlee.018784.0008]          \
database   : [mongodb://singharoy:****@129.114.17.185:27017/mdffentk]         ok
create pilot manager                                                          ok
submit 1 pilot(s)
        pilot.0000   xsede.bridges2          128 cores       0 gpus           ok

```

There will be waiting first (looks like the screen is frozen but actually the
HPC job is in the queue and trying to get it scheduled on available compute
nodes in the cluster) and the progress will be visible once HPC job is active,
like:

```
...
Update: DeepDriveMD.MolecularDynamics state: SCHEDULING
Update: DeepDriveMD.MolecularDynamics.task.0031 state: SCHEDULING
Update: DeepDriveMD.MolecularDynamics.task.0031 state: SCHEDULED
submit: ########################################################################
Update: DeepDriveMD.MolecularDynamics state: SCHEDULED
Update: DeepDriveMD.MolecularDynamics.task.0031 state: SUBMITTING
Update: DeepDriveMD.MolecularDynamics.task.0031 state: EXECUTED
Update: DeepDriveMD.MolecularDynamics state: DONE
Update: DeepDriveMD.MachineLearning state: SCHEDULING
Update: DeepDriveMD.MachineLearning.task.0032 state: SCHEDULING
Update: DeepDriveMD.MachineLearning.task.0032 state: SCHEDULED
Update: DeepDriveMD.MachineLearning state: SCHEDULED
submit: ########################################################################
Update: DeepDriveMD.MachineLearning.task.0032 state: SUBMITTING
Update: DeepDriveMD.MachineLearning.task.0032 state: EXECUTED
Update: DeepDriveMD.MachineLearning state: DONE
Update: DeepDriveMD.ModelSelection state: SCHEDULING
Update: DeepDriveMD.ModelSelection.task.0033 state: SCHEDULING
Update: DeepDriveMD.ModelSelection.task.0033 state: SCHEDULED
Update: DeepDriveMD.ModelSelection state: SCHEDULED
submit: ########################################################################
Update: DeepDriveMD.ModelSelection.task.0033 state: SUBMITTING
Update: DeepDriveMD.ModelSelection.task.0033 state: EXECUTED
Update: DeepDriveMD.ModelSelection state: DONE
Update: DeepDriveMD.Agent state: SCHEDULING
Update: DeepDriveMD.Agent.task.0034 state: SCHEDULING
Update: DeepDriveMD.Agent.task.0034 state: SCHEDULED
Update: DeepDriveMD.Agent state: SCHEDULED
submit: ########################################################################
Update: DeepDriveMD.Agent.task.0034 state: SUBMITTING
Update: DeepDriveMD.Agent.task.0034 state: EXECUTED
Update: DeepDriveMD.Agent state: DONE
Finishing stage 3 of 4
...
```

# bridges2 job commands

* Running jobs to check:

```
$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1484680        RM pilot.00    hrlee  R       1:36      1 r138
```

* Canceling jobs:

```
scancel <JOBID>
scancel 1484680
``` 
# EnTK specific commands

DeepDriveMD is running based on EnTK, ensemble toolkit, which creates an unique
session id to store necessary data and logs for each run. For example, `python
-m deepdrivemd.deepdrivemd xx.yaml` will leave a sub-directory like:

* Session directory (temporal, auto-generated):

```
re.session.br013.ib.bridges2.psc.edu.hrlee.018784.0008
```

This session data is useful for further analysis, and can be removed if runs are complete. 

* Check debugging messages:

```
more <session id>/radical.log
```

* Check task outputs and error messages:
Individual has `radical.pilot.sandbox` to keep all the output and error
messages of DeepDriveMD tasks. For example, OpenMM python script and pytorch
script may generate stdout/stderr messages and all these output of a specific
run are stored under the sandbox directory.

```
/ocean/projects/mcb110096p/hrlee/radical.pilot.sandbox/
```

# PSC SSH

Please refer to https://www.psc.edu/about-using-ssh/

