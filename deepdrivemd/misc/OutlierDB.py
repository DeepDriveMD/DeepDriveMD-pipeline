import subprocess
import time
import os
import sys
import random
import math

import MDAnalysis as mda
from  MDAnalysis.analysis.rms import RMSD


class OutlierDB:
    def __init__(self, dir, restarts):
        self.dir = dir
        self.sorted_index = list(map(lambda x: os.path.basename(x[1]).replace(".pdb",""), restarts ))
        self.dictionary = {}
        for rmsd, path in restarts:
            md5 = os.path.basename(path).replace(".pdb","")
            self.dictionary[md5] = [rmsd, 0]
        self.print()
        self.test_increasing()

    def print(self, n=100):
        print("="*30)
        print("In OutlierDB")
        for i in range(n):
            md5 = self.sorted_index[i]
            print(f"{md5}: {self.dictionary[md5]}")
        print("="*30)

    def test_increasing(self):
        for i in range(len(self.sorted_index)-1):
            md5a = self.sorted_index[i]
            md5b = self.sorted_index[i+1]
            if(self.dictionary[md5b] < self.dictionary[md5a]):
                print(f"OutlierDB: not increasing ({md5a}: {self.dictionary[md5a]}) - ({md5b}: {self.dictionary[md5b]})") 
                return False
        print("OutlierDB: increasing")
        return True

    def is_consistent(self):
        print("="*30)
        print("OutlierDB: is_consistent")
        for md5 in self.sorted_index:
            stored_rmsd = self.dictionary[md5][0]
            pdb_file = f"{self.dir}/outlier_pdbs/{md5}.pdb"
            print(f"OutlierDB: {pdb_file}")
            if(not os.path.exists(pdb_file)):
                print(f"OutlierDB: {pdb_file} does not exist")
            else:
                top_dir = self.dir + "/.."
                outlier = mda.Universe(pdb_file)
                ref = mda.Universe(f'{top_dir}/MD_exps/fs-pep/pdb/1FME.pdb')
                R = RMSD(outlier, ref, select='protein and name CA')
                R.run()
                computed_rmsd = R.rmsd[0,2]
                print(f"OutlierDB: stored_rmsd = {stored_rmsd}, computed_rmsd = {computed_rmsd}")
        print("="*30)

    def update(self, restarts): 
        self.sorted_index = list(map(lambda x: os.path.basename(x[1]).replace(".pdb",""), restarts ))
        for rmsd, path in restarts:
            md5 = os.path.basename(path).replace(".pdb","")
            if(md5 in self.dictionary): 
                old_rmsd, n = self.dictionary[md5]
                n += 1
                if(old_rmsd != rmsd):
                    print(f"old_rmsd = {old_rmsd}, rmsd = {rmsd}")
                self.dictionary[md5] = [rmsd, n]
            else:
                self.dictionary[md5] = [rmsd, 0]
        dkeys = list(self.dictionary.keys())
        for k in dkeys:
            if not (k in self.sorted_index):
                del self.dictionary[k]

    def increment(self, path):
        md5 = os.path.basename(path).replace(".pdb","")
        self.dictionary[md5][1] += 1

    def next(self):
        if(len(self.sorted_index) == 0):
            print("Bug")
            return None
        print(f"In next: len(self.sorted_index) = {len(self.sorted_index)}")
        for md5 in self.sorted_index:
            print(f"self.dictionary[{md5}] = {self.dictionary[md5]}")
            if(self.dictionary[md5][1] == 0): #unused
                print("Here?")
                self.dictionary[md5][1] += 1
                return f"{self.dir}/outlier_pdbs/{md5}.pdb"
        md5 = self.sorted_index[0]
        self.dictionary[md5][1] += 1
        return f"{self.dir}/outlier_pdbs/{md5}.pdb"

    def next_random(self, m = None):
        if(len(self.sorted_index) == 0):
            print("Bug")
            return None
        if(m == None):
            hlimit = len(self.sorted_index) - 1
        else:
            hlimit = min(m, len(self.sorted_index) - 1)
        i = int(random.betavariate(alpha=1, beta=25)*(hlimit))
        md5 = self.sorted_index[i]

        selected_rmsd = self.dictionary[md5][0]
        rmsds = list(map(lambda x: x[0], self.dictionary.values()))
        min_rmsd = min(rmsds)
        max_rmsd = max(rmsds)
        print(f"In next_random: selected_rmsd = {selected_rmsd}, min_rmsd = {min_rmsd}, max_rmsd = {max_rmsd}, md5 = {md5}, index = {i}, len = {len(self.sorted_index)}") 

        return f"{self.dir}/outlier_pdbs/{md5}.pdb"

    def next_best(self):
        md5 = self.sorted_index[0]
        selected_rmsd = self.dictionary[md5][0]
        rmsds = list(map(lambda x: x[0], self.dictionary.values()))
        min_rmsd = min(rmsds)
        max_rmsd = max(rmsds)
        print(f"In next_best: selected_rmsd = {selected_rmsd}, min_rmsd = {min_rmsd}, max_rmsd = {max_rmsd}, md5 = {md5}, len = {len(self.sorted_index)}")
        return f"{self.dir}/outlier_pdbs/{md5}.pdb"

    def next_10best(self):
        n = 20
        if(len(self.sorted_index) - 1 < n):
            n = len(self.sorted_index) - 1
        i = int(random.random()*n)
        md5 = self.sorted_index[i]
        selected_rmsd = self.dictionary[md5][0]
        rmsds = list(map(lambda x: x[0], self.dictionary.values()))
        min_rmsd = min(rmsds)
        max_rmsd = max(rmsds)
        print(f"In next_best: selected_rmsd = {selected_rmsd}, min_rmsd = {min_rmsd}, max_rmsd = {max_rmsd}, md5 = {md5}, index = {i}, len = {len(self.sorted_index)}")
        return f"{self.dir}/outlier_pdbs/{md5}.pdb"


def my_lock(fn, sleeptime=3):
    fn_lock = fn + ".lock"
    while(os.path.exists(fn_lock)):
        t = random.random()*sleeptime
        print(f"sleeping in lock {fn_lock} for {t} at {time.asctime()}"); sys.stdout.flush()
        time.sleep(t)
    subprocess.getstatusoutput(f'touch {fn_lock}')
    print(f"Locking {fn_lock} at {time.asctime()}")

def my_unlock(fn):
    fn_lock = fn + ".lock"
    print(f"Unlocking {fn_lock} at {time.asctime()}"); sys.stdout.flush()
    subprocess.getstatusoutput(f'rm -f {fn_lock}')
