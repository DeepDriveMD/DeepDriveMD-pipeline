import time
import random
import sys

i = 0
while(True):
    print(f"Iteration = {i}")
    sleeptime = random.randint(5, 20)
    print(f"  Sleeping for {sleeptime}")
    time.sleep(sleeptime)
    sys.stdout.flush()
    i += 1
