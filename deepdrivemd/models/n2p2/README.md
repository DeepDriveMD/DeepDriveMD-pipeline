# N2P2

N2P2 is a neural network potential for molecular dynamics force fields.
The force field needs to be accessed through LAMMPS[^1].

The N2P2 code[^2] builds as separate package. The build process also 
automatically builds LAMMPS with N2P2 builtin. However, we cannot use
that version of LAMMPS here as we need LAMMPS to produce trajectory files
in the DCD format. So we need to copy the USER-NNP package into the LAMMPS
source directory. Subsequently we need to address an update to the LAMMPS
API. In LAMMPS the request members `pair`, `half`, and `full` are now protected.
This means that their values cannot be changed directly any longer. As a result 
the code in `USER-NNP/pair_nnp.cpp` function `void PairNNP::init_style()`
needs to be changed like
```C++
  /*
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  */
  neighbor->requests[irequest]->enable_full();
```
The N2P2 workflow is a bit different than, e.g. the DeePMD workflow. DeePMD
just requires you to provide the training data and then `dp train` will 
essentially train the model for you. In N2P2 more steps are needed:

- sfparamgen  - generate the symmetry functions
- nnp-scaling - calculates the symmetry functions
- nnp-train   - does the actual training

[^1]: "LAMMPS packages" https://www.lammps.org/external.html
[^2]: "n2p2 - A neural network potential package" https://github.com/CompPhysVienna/n2p2
