# Post-production

This directory contains scripts to analyze the results from a DeepDriveMD simulation.
There is also a `postproduction_stream` directory that contains similar scripts for 
the streaming version of DeepDriveMD, which uses ADIOS2 for handling data. Here we do
not consider results from the streaming version.

## Production

Assuming that you have DeepDriveMD properly installed along with any external packages
needed for your simulation, e.g. OpenMM or NWChem, you should be able to run some
of the cases in the `test` directory. The "original" test case is the `1MFE` one
that was used for the protein folding paper. You should be able to run this test case
using

```
cd DeepDriveMD-pipeline/test
nohup make run12 2>&1 > run12.out &
```

At the end of this simulation you should have a number of directories:

- In the test directory there will be a client session directory with the
  name `re.session.<machine_name>.<user_name>.*` containing data pertaining
  to the workflow execution.
- In the scratch directory there will be pilot session directory with name
  `radical.pilot.sandbox/re.session.<machine_name>.<user_name>.*` containing
  additional data about the workflow execution from the pilot perspective.
- Finally there is the test directory with the scientific data. For NWChem
  examples this directory is called `test_nwchem`.

## Visualizing scientific results

For the test cases there are two structures for every chemical system:

- The reference structure is stored in a PDB file 
  `DeepDriveMD-pipeline/data/<system>/<name>-folded.pdb`. For NWChem there
  might be a special PDB file because that application tends to add, rename,
  and reorder atoms. Therefore a PDB file straight from the Protein Databank 
  will not match the chemical structure that NWChem is actually simulating.
- The starting structure for the simulation is a PDB file 
  `DeepDriveMD-pipeline/data/<system>/system/<name>-unfolded.pdb`.

Molecular dynamics simulations are run the results stored under the directory
`test_nwchem/molecular_dynamics_runs/stage*/task*`. Key files are 

- `stage*_task*.dcd` which is the trajectory in the DCD format.
- `stage*_task*.h5` which is an HDF5 file containing the contact maps between
  the selected atoms, the selected atom positions, and the RMSD relative to
  the reference structure.

The HDF5 file contains all information necessary to map structures from the
trajectory into the latent space of the ML model that drives the dynamics.

The machine learning models are stored in the directory
`test_nwchem/machine_learning_runs/stage*/task*/checkpoint/epoch-*-<date>-<time>.h5`.
Note that all the models that were generated throughout the simulation are kept.
Nevertheless the last model is the only one that really matters.

Here we are interested in producing graphical representations of the simulation 
results as shown in Figure 5 of [Bhowmik 2018](#bhowmik-2018). 
In order to generate such graphics we need:

- The ML model
- The HDF5 data files
- 2 or 3 selected coordinates from the latent space
- The name of the output file

With this data we proceed to generate the data to go into the plotting routine:

  1. Read the ML model
  2. Separate the encoder
  3. Iterate over the HDF5 files
     1. Iterate over steps
        1. Extract the contact map
        2. Extract the RMSD
        3. Infer the latent space coordinates from the contact map
        4. Select the desired coordinates from the latent space
        5. Write the selected coordinates and the RMSD value to a file

The plotting routine itself needs:

- The name of the data file
- Any relevant plotting parameters

With this data we proceed to:

  1. Build a plotting context
  2. Read the data in
  3. Add the data to the plot
  4. Show or save the plot

## Visualizing performance data

DeepDriveMD executes the workflow using the RADICAL-Cybertools. These tools
both execute the workflow but also collect performance related data and store
those data in the "client sandbox". I have pulled a Python script together
based on Mikhail Titov's Jupyter Notebook "radical-plotting.ipynb" and 
called it (unimaginatively) `radical-plotting.py`.

## References

<A name="bhowmik-2018">[Bhowmik 2018]</A>
    Bhowmik, D., Gao, S., Young, M.T. et al. 
    "Deep clustering of protein folding simulations". _BMC Bioinformatics_ **19**
    (Suppl 18), 484 (2018). https://doi.org/10.1186/s12859-018-2507-5
