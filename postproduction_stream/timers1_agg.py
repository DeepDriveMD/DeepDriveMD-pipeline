import glob
import sys
import subprocess
import os

print(subprocess.getstatusoutput("hostname")[1])

pattern = sys.argv[1]
output_dir = sys.argv[2]

print(pattern)

csv_files = " ".join(glob.glob(pattern))

print("csv_files = ", csv_files)

dir = "/p/gpfs1/yakushin/Outputs/" + output_dir + "/postproduction"

try:
    os.mkdir(dir)
except Exception as e:
    print(e)


outfile = f"{dir}/timers.csv"

print(outfile)

print(
    subprocess.getstatusoutput(
        f'echo ",label,start,gps,date,file,line,unit,time" > {outfile}'
    )
)
print(subprocess.getstatusoutput(f"cat {csv_files} | grep -v unit,time  >> {outfile}"))
