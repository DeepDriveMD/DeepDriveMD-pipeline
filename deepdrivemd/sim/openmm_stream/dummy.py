import time
import random

import shutil
import argparse
from pathlib import Path
from typing import Optional
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
#from mdtools.openmm.sim import configure_simulation
#from mdtools.openmm.reporter import OfflineReporter
from deepdrivemd.utils import Timer
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.sim.openmm.config import OpenMMConfig


#import simtk.unit as u
import sys, os, shutil 
import argparse 
import time
import subprocess
#from OutlierDB import *
import pickle
#import mytimer
from lockfile import LockFile





i = 0
while(True):
    print(f"Iteration = {i}")
    sleeptime = random.randint(5, 20)
    print(f"  Sleeping for {sleeptime}")
    time.sleep(sleeptime)
    sys.stdout.flush()
    i += 1
