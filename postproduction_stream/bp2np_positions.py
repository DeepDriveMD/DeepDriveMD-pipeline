import numpy as np
import adios2
import sys

fn = sys.argv[1]
output = fn.replace("trajectory.bp", "positions.npy")

with adios2.open(fn, "r") as fr:
    n = fr.steps()
    shape = list(
        map(
            int, fr.available_variables()["positions"]["Shape"].replace(",", "").split()
        )
    )
    positions = fr.read("positions", [0, 0], shape, 0, n)
    np.save(output, positions)
