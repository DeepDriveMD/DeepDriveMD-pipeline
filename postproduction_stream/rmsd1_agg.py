import glob
import sys
import subprocess
import os

print(subprocess.getstatusoutput("hostname")[1])

pattern = sys.argv[1]

print(pattern)

csv_files = " ".join(glob.glob(pattern))

print("csv_files = ", csv_files)

dir = "/".join(pattern.split("/")[:-5]) + "/postproduction"

try:
    os.mkdir(dir)
except Exception as e:
    print(e)


outfile = f"{dir}/rmsd.csv"

print(outfile)

print(subprocess.getstatusoutput(f'echo "gpstime,rmsd" > {outfile}'))
print(subprocess.getstatusoutput(f"cat {csv_files} | grep -v gps | sort >> {outfile}"))
