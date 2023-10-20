# 7CZ4 - NSP3 macro domain with bound ligand

The structures stored in this directory relate to the ligand binding use case.
The original structure was obtained from the protein databank ID
[7CZ4](https://www.rcsb.org/structure/7cz4).
The initial structure was incomplete missing some heavy atoms as well 
as Hydrogens. This structure was "fixed" with 
[Moprobity](http://molprobity.biochem.duke.edu/) and
[PDBFixer](https://github.com/openmm/pdbfixer). 
Finally, only the monomer of the protein was kept. The resulting structure
is stored in `7CZ4-folded.pdb`. This name is analogous to the way systems 
are named in the `bba` directory. 
More details on how this structure was prepared can be found at
<https://github.com/hjjvandam/nwchem-1/tree/pretauadio2/QA/tests/7cz4>.

The structure in `system/7CZ4-unfolded.pdb` was created by first shifting
the ligand out of the protein, and subsequently running dynamics on it at 
310K. Conventional dynamics was not able to have the ligand find its binding
location.
