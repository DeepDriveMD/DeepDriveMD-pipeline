import argparse
from pathlib import Path
import subprocess
import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_dir",
    "-o",
    help="a number corresponding to a directory in /p/gpfs1/yakushin/Outputs",
    type=int,
    default=3,
)

parser.add_argument("--session", "-s")


args = parser.parse_args()

output_dir = Path(f"/p/gpfs1/yakushin/Outputs/{args.output_dir}/logs")
output_dir.mkdir(parents=True, exist_ok=True)

log = glob.glob("../*.log")[0]
subprocess.getstatusoutput(f"cp {log} {str(output_dir)}")

log = f"../{args.session}"
subprocess.getstatusoutput(f"rsync -a {log} {str(output_dir)}")

output_dir = Path(f"/p/gpfs1/yakushin/Outputs/{args.output_dir}/logs/sandbox")
output_dir.mkdir(parents=True, exist_ok=True)

log = f"/p/gpfs1/yakushin/radical.pilot.sandbox/{args.session}"
subprocess.getstatusoutput(f"rsync -a {log} {str(output_dir)}")
