import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    "-o",
    help="a number corresponding to a directory in /p/gpfs1/yakushin/Outputs",
    type=int,
    default=3,
)
parser.add_argument("--nodes", "-n", help="number of nodes to use", type=int, default=1)
parser.add_argument(
    "--walltime", "-w", help="walltime in minutes", type=int, default=10
)

args = parser.parse_args()

print(args)
print(dir(args))
