This directory contains two sets of files

- PDB file(s) with the structure of the formaldehyde isomer
- NWChem input files for specific training data points

The input files target training data points that are difficult
to generate automatically. The structures are radicals with particular
spin states. These are easily to set up by hand but automatically
guessing the correct spin state would require non-trivial script
writing. So, it is easier to provide the finished input files.

The numbering of the diatomic input files has the following meaning:

- 000000 - the equilibrium structure
- 00000n - structures with elongated bonds relative to the equilibrium structure,
           the higher n the more elongated the bond
- 00001n - structures with shortened bonds relative to the equilibrium structure,
           the higher n the shorter the bond

Because the energy rises much faster for shortened bonds than for elongated bonds,
the shortening happens in small step (in most cases 0.1 Angstrom), whereas the
elongating happens in increasingly larger steps (up to a maximum of 2 Angstrom
at the moment).

The hope is that by providing training data points for individual atoms and 
diatomic systems we can "anchor" the neural network potential to the behavior 
of well understood sub-systems. In addition the corresponding data points
are easily calculated.
