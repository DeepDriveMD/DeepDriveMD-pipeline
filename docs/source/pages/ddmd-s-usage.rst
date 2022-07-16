======================================================================================
How to use DeepDriveMD-S on Lassen@LLNL with the existing installation of dependencies 
======================================================================================

---
Run
---

#. Set up the environment::

     module load gcc/7.3.1
     . /etc/profile.d/conda.sh
     conda activate /usr/workspace/cv_ddmd/conda1/powerai
     export IBM_POWERAI_LICENSE_ACCEPT=yes
     module use /usr/workspace/cv_ddmd/software1/modules
     module load adios2

   Usually I source something like `powerai.sh <https://github.com/DeepDriveMD/DeepDriveMD-pipeline/blob/develop/bin/powerai.sh>`_.
     
#. For convenience, let us define::

     export DDMD="/usr/workspace/cv_ddmd/$USER/DeepDriveMD-pipeline"

#. To run various preconfigured examples::

     cd $DDMD/test
     nohup make X > X.log 2>&1 &

   where ``X`` is one of these:

   * ``run1`` - mini BBA run with 12 simulations, 1 aggregator; 30m.
   * ``run2`` - production BBA run with 120 simulations, 10 aggregators; 12h.
   * ``run3`` - production BBA run, no outliers, no machine learning is used,
     states for the next batch of the simulations are selected randomly from the reported states; 120 simulations, 10 aggregators; 12h.
   * ``run3m`` - same as ``run3``, but 12 simulations, 1 aggregator; 30m.
   * ``run4`` - production BBA run, random selection of outliers; 120 simulations, 10 aggregators; 12h.
   * ``run4m`` - same as ``run4`` by 12 simulations, 1 aggregator; 30m.
   * ``run5`` - production BBA run, greedy selection from the traversed states based on RMSD, no outliers are used; 120 simulations, 10 aggregators; 12h.
   * ``run5m`` - same as ``run5`` but 12 simulations, 1 aggregator; 30m.
   * ``run6`` - same as ``run9`` but 12 simulations, 1 aggregator; 3h.
   * ``run6a`` - same as ``run9a`` but 12 simulations, 1 aggregator, 3h.
   * ``run7`` - production run for insRec_OM_region; 120 simulations, 10 aggregators; 12h.
   * ``run7m`` - same as run7 but 12 simulations, 1 aggregator; 4h.
   * ``run8`` - production run for spike; 120 simulations, 10 aggregators; 12h.
   * ``run9`` - production run for smoothended_rec, 120 simulations, 10 aggregators; 12h.
   * ``run9a`` - production run for smoothended_rec where CVAE is replaced by AAE; 120 simulations, 10 aggregators; 12h.
   * ``run10`` - same as run11 but 12 simulations, 1 aggregator; 6h.
   * ``run11`` - production run for multi-ligand case; 120 simulations, 10 aggregators; 12h.

#. The configuration files for the above cases can be found by appending ``$DDMD`` with ``test/bba/DIR``
   where ``DIR`` is listed in the ``Makefile``. For example, for run1, the path is ``$DDMD/test/bba/test1_stream``.

#. DeepDriveMD is configured using a YAML file, we edit ``generate.py`` to generate the YAML file (make sure to activate 
   the conda environment before running this)::

     python generate.py > config.yaml

#. ``adios_file.xml`` controls the output of a complete trajectory including positions, velocities, contact maps from each simulation.

#. ``adios_sim.xml`` controls the communication over the network between a simulation and an aggregator.

#. ``adios_agg_4ml.xml`` controls the communication over the network between an aggregator and training.

#. ``adios_agg.xml`` controls the communication over the network between an aggregator and inference.

#. Notice that ``bin/run.sh`` command sets up authorization in this line::

     source /usr/workspace/cv_ddmd/.radical/auth

   For the content of this file for a particular cluster, ask Radical developers.
   In this file various environmental variables are set, such as RMQ_HOSTNAME, RMQ_PASSWORD,
   mongodb_host, RADICAL_PILOT_DBURL, ..., that allow Radical Ensemble Toolkit to
   communicate with the corresponding servers.

-------
Results
-------

#. Radical logs: ``/p/gpfs1/$USER/radical.pilot.sandbox``
#. The latest best model: ``{experiment_directory}/machine_learning_runs/stage0000/task0000/published_model/best.h5``, where ``{experiment_directory}``
   is specified in ``generate.py`` (and in the corresponding ``config.yaml``)
#. ADIOS trajectories from each simulation invocation: ``{experiment_directory}/molecular_dynamics_runs/stage0000/task*/*/trajectory.bp``

   * There is only one stage and as many tasks under ``molecular_dynamics_runs/stage0000`` directory as there are parallel simulations
     (in our typical production run we use 120 parallel simulations, in mini test runs - 12 simulations).
   * Under each ``taskXXXX`` directory, there are subdirectories 0, 1, ... corresponding to different restarts of the simulations from the outliers.
   * The first one or two simulations are started from initial conditions. For the rest, the corresponding outlier files are copied into the directory.
   
--------------
Postproduction
--------------

^^^^^^^^^^^^
Trajectories
^^^^^^^^^^^^

#. Trajectories in bp format are saved for each simulation. For example::

     $ bpls /p/gpfs1/yakushin/Outputs/3/molecular_dynamics_runs/stage0000/task0000/4/trajectory.bp
       char     contact_map  100*{28, 28}
       int32_t  gpstime      100*{1}
       int64_t  md5          100*{32}
       float    positions    100*{504, 3}
       float    rmsd         100*{1}
       int32_t  step         100*{1}

#. The above output says that there are 100 time steps saved in ``trajectory.bp``.
   For each time step, we save 28x28 contact map, gpftime when the time step was reported,
   md5 sum of positions, positions (in this case, it is x,y,z coordinates for each of the 504 atoms, corresponding
   velocities, rmsd to the folded state, reporting step in the simulation (here it is from 0 to 99).
#. To convert those trajectories into npy format, using 4 nodes, 10-minute walltime, do::

     cd $DDMD/postproduction_stream
     nohup ./run_positions.py 3 4 10 > positions_3.log 2>&1 &

   Here 3 corresponds to the output subdirectory for the run: ``/p/gpfs1/yakushin/Outputs/3``.
#. Running the above command creates `positions.npy` in each directory where `trajectory.bp` is found.
#. The script uses Radical-ENTK to create as many independent tasks (that can run in parallel) as there are trajectories.
#. You might have to edit ``driver_positions.py`` to change the path to your python and to your ``$DDMD``, your file that sets the environment for the jobs.

^^^^^^^^^^^
Loss curves
^^^^^^^^^^^

#. To generate loss curves from logs, run, for example::

     python loss_real1.py -s re.session.lassen709.yakushin.019150.0009 -p 0 -t 13

   provided that the log file for the machine learning task is in::

     /p/gpfs1/$USER/radical.pilot.sandbox/re.session.lassen709.yakushin.019150.0009\
     /pilot.0000/task.0013/task.0013.out

   * The corresponding ``*.csv`` file will be in ``/p/gpfs1/$USER/Outputs/3/postproduction_stream/losses.csv``.
   * The loss curves can be plotted by with ``$DDMD/postproduction_stream/loss.ipynb``.

#. For AAE case, the logs are written in a different format and can be parsed with::

     python loss_aae.py logfile dir

   where ``dir`` is subdirectory of ``/p/gpfs1/yakushin/Outputs`` where the run files are written.

^^^^^^^^^^^^
Gantt charts
^^^^^^^^^^^^

#. To parse log files in order to generate gantt charts, run from postproduction_stream directory::

     nohup ./run_timers.sh output_dir nodes walltime session pilot exclude > timers.log 2>&1 &

   where ``output_dir`` - subdirectory of ``/p/gpfs1/yakushin/Outputs``, ``nodes`` - number of nodes to use (the job is submitted to the cluster),
   ``walltime`` - maximum walltime, ``session`` - log session like ``re.session.lassen709.yakushin.019150.0009``, ``pilot`` - typically 0, ``exclude`` - what tasks to exclude (for example, it is not interesting to see timing
   for aggregators and it takes a lot of time to parse the corresponding logs so you can exclude those by using ``120-129`` as exclude (currently it is dumb and just parses one range).

#. An example of the notebook to plot gantt charts: ``postproduction_stream/gantt_rmsd_streaming.ipynb``


^^^^^^^^^^
Embeddings
^^^^^^^^^^

#. To generate embedding files, run from the postproduction_stream directory::

     nohup ./run_emb.sh > emb.log outputdir nodes walltime zcentroid 2>&1 &

   where ``outputdir`` - subdirectory of ``/p/gpfs1/yakushin/Outputs/`` where job outputs are stored, ``nodes`` - number of nodes to use (the job is submitted to the cluster),
   ``walltime`` - up to how much time to run the job, ``zcentroid`` - 1 or 0 depending on whether you want to calculate zcentroid or not.

#. An example notebook to plot embeddings: ``postproduction_stream/plot_tsne.ipynb``

^^^^^^^^^
Positions
^^^^^^^^^

#. To generate positions in numpy format, run from the postproduction_stream directory::

     nohup ./run_positions.sh output_dir nodes walltime > positions.log 2>&1 &

#. To collect the results into a single file, run from ``/p/gpfs1/yakushin/Outputs`` directory::

     /path/postproduction_stream/archive.sh outputdir


   
.. autosummary::
    :toctree: _autosummary
    :recursive:

    deepdrivemd

