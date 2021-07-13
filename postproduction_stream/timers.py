import glob
import subprocess
import pandas as pd
import os
import sys

dir = sys.argv[1]

session = os.path.basename(os.path.dirname(dir))

#session = subprocess.getstatusoutput('ls -d ../re.* | cut -d "/" -f2')[1]
#dir = f'/p/gpfs1/yakushin/radical.pilot.sandbox/{session}/pilot.0000'

stdouts = glob.glob(f'{dir}/task*/*.out')

pf = pd.DataFrame(columns = ["label", "start", "gps", "date", "file", "line", "unit", "time"])

labels = []
starts = []
gpss = []
dates = []
files = []
nlines = []
units = []
times = []



fn = f"{session}.csv"

for s in stdouts:
    unit = int(s.split("/")[-2].replace("task.",""))
    with open(s) as f:
        lines = list(filter(lambda x: x.find("TLaBeL") == 0 and x.find("Testing") == -1, f.readlines()))
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


print(len(labels))


pf['label'] = labels
pf['start'] = starts
pf['gps'] = gpss
pf['date'] = dates
pf['file'] = files
pf['line'] = nlines
pf['unit'] = units
pf['time'] = times

pf.to_csv("/tmp/" + fn)

