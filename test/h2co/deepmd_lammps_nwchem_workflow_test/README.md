# DeePMD-LAMMPS-NWChem workflow

This workflow combines ab-initio calculations, neural network potentials,
and molecular dynamics to tackle a quantum chemical problem with DeepDriveMD.

Requirements for this workflow to work are the same as the usual ones for
DeepDriveMD. In addition 

* NWChem is needed for the ab-initio calculations,
* the Atomic Simulation Environment (ASE) is needed to move data between 
NWChem, DeePMD, and LAMMPS, 
* DeePMD is needed for the neural network potentials (both training and evaluation)
* LAMMPS is needed to run the MD simulations using the neural network potentials
* TensorFlow is needed to provide the machine learning infrastructure.
