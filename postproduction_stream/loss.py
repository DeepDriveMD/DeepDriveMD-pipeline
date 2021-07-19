import numpy as np
import sys
import pandas as pd

log = sys.argv[1]

with open(log) as f:
    lines = list(
        filter(lambda x: x.find("val_loss") >= 0 and x.find("===") >= 0, f.readlines())
    )
    losses = list(
        map(lambda x: list(map(float, np.array(x.strip().split(" "))[[7, 10]])), lines)
    )

df = pd.DataFrame(columns=["train_loss", "val_loss"])

df["train_loss"] = list(map(lambda x: x[0], losses))
df["val_loss"] = list(map(lambda x: x[1], losses))

df.to_csv("/tmp/losses.csv")
