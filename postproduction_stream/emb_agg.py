import glob
import sys
import subprocess
import os
import numpy as np
from pathlib import Path

from openTSNE import TSNE

print(subprocess.getstatusoutput("hostname")[1])

pattern = sys.argv[1]

print(pattern)

dirs = glob.glob(pattern)
dirs.sort()


output_dir = Path("/".join(pattern.split("/")[:-5]) + "/postproduction/embeddings")

output_dir.mkdir(parents=True, exist_ok=True)

emb = []
rmsd = []

for d in dirs:
    if os.path.exists(d + "/embeddings_cvae.npy") and os.path.exists(d + "/rmsd.npy"):
        emb.append(np.load(d + "/embeddings_cvae.npy"))
        rmsd.append(np.load(d + "/rmsd.npy"))

embeddings = np.concatenate(emb)

np.save(f"{str(output_dir)}/embeddings.npy", embeddings)
np.save(f"{str(output_dir)}/rmsd.npy", np.concatenate(rmsd))

print("Finished aggregation")
sys.stdout.flush()

tsne2 = TSNE(n_components=2, n_jobs=39)
trained = tsne2.fit(embeddings)
print("type(trained)=", type(trained))
tsne_embeddings2 = trained.transform(embeddings)

with open(f"{str(output_dir)}/tsne_embeddings_2.npy", "wb") as f:
    np.save(f, tsne_embeddings2)

print("Finished 2D TSNE")
sys.stdout.flush()

"""
tsne3 = TSNE(n_components=3, n_jobs=39)
trained = tsne3.fit(embeddings)
print("type(trained)=", type(trained))
tsne_embeddings3 = trained.transform(embeddings)

with open(f"{str(output_dir)}/tsne_embeddings_3.npy", "wb") as f:
    np.save(f, tsne_embeddings3)


print("Finished 3D TSNE"); sys.stdout.flush()

"""
