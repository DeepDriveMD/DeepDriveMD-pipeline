import glob
from collections import Counter

outs = glob.glob("task.*/*.out")
outs.sort()

b = []

for o in outs[:-3]:
    # print(o); sys.stdout.flush()
    with open(o) as f:
        a = list(
            map(
                lambda y: int(y.split(" ")[3].replace(",", "")),
                filter(lambda x: x.find("init_multi_ligand") == 0, f.readlines()),
            )
        )
        # print(a); sys.stdout.flush()
        b.append(a)


print(b)

for i in range(3):
    c = Counter(map(lambda x: x[i], b))
    print(c)
    print(len(c))


"""
(/usr/workspace/cv_ddmd/conda1/powerai) [yakushin@lassen708:task.0000]$ grep "init_multi_ligand: id" ../task.0011/*.out
init_multi_ligand: id = 11, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1266-123700/system/sys_l1266-123700.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1266-123700
init_multi_ligand: id = 3, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1100-989050/system/sys_l1100-989050.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1100-989050
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
init_multi_ligand: id = 5, pdb = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890/system/sys_l1102-1044890.pdb, tdir = /usr/workspace/cv_ddmd/yakushin/Integration1/data/ml/l1102-1044890
"""
