import glob
import subprocess

# iteration_3_ligand_116_natoms_132163_sim_16_agg_1.npy

fs = glob.glob("*.npy")

for f in fs:
    tokens = f.split("_")
    if tokens[3] == "-1":
        tokens[3] = tokens[7]
        nf = "_".join(tokens)
        print(f"{f} -> {nf}")
        subprocess.getstatusoutput(f"mv {f} {nf}")
