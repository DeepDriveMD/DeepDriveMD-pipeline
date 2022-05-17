import glob
import pandas as pd
import os
import sys
import argparse
import subprocess

print(subprocess.getstatusoutput("hostname")[1])
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--session",
    "-s",
    help="re.session*",
)
parser.add_argument("--pilot", "-p", help="pilot", type=int, default=0)
parser.add_argument("--task", "-t", help="task id", type=int, default=0)
# parser.add_argument("--output_dir", "-o", help="output directory number", type=int, default=3)

args = parser.parse_args()

top_logs_dir = "/p/gpfs1/yakushin/radical.pilot.sandbox/"
pilot = f"pilot.{args.pilot:04d}"
task = f"task.{args.task:04d}"
log_pattern = top_logs_dir + "/" + args.session + "/" + pilot + "/" + task + "/*.out"
print(log_pattern)
log = glob.glob(log_pattern)[0]
out = os.path.dirname(log) + "/timers.csv"

pf = pd.DataFrame(
    columns=["label", "start", "gps", "date", "file", "line", "unit", "time"]
)

labels = []
starts = []
gpss = []
dates = []
files = []
nlines = []
units = []
times = []

for s in [log]:
    unit = int(s.split("/")[-2].replace("task.", ""))
    with open(s) as f:
        lines = list(
            filter(
                lambda x: x.find("TLaBeL") == 0 and x.find("Testing") == -1,
                f.readlines(),
            )
        )
        for line in lines:
            tokens = line.split("|")
            labels.append(tokens[1])
            starts.append(int(tokens[2]))
            gpss.append(int(float(tokens[3])))
            dates.append(tokens[4])
            files.append(os.path.basename(tokens[5]))
            nlines.append(int(tokens[6].strip()))
            units.append(unit)
            times.append(float(tokens[7].strip()))


print("len(labels) = ", len(labels))
sys.stdout.flush()
sys.stderr.flush()


pf["label"] = labels
pf["start"] = starts
pf["gps"] = gpss
pf["date"] = dates
pf["file"] = files
pf["line"] = nlines
pf["unit"] = units
pf["time"] = times

pf.to_csv(out)
