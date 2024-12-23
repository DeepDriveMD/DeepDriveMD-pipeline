import deepmd
import os
from pathlib import Path

cwd = os.getcwd()
data_path = Path(cwd,"../../sim/nwchem/test_dir")
json_file = Path(cwd,"input.json")
train1 = Path("./train-1")
train2 = Path("./train-2")
train3 = Path("./train-3")
train4 = Path("./train-4")
ckpt = Path("model.ckpt")

deepmd.gen_input(data_path,json_file)
if not train1.exists():
    deepmd.train(train1,json_file)
else:
    deepmd.train(train1,json_file,ckpt_file=ckpt)

if not train2.exists():
    deepmd.train(train2,json_file)
else:
    deepmd.train(train2,json_file,ckpt_file=ckpt)

if not train3.exists():
    deepmd.train(train3,json_file)
else:
    deepmd.train(train3,json_file,ckpt_file=ckpt)

if not train4.exists():
    deepmd.train(train4,json_file)
else:
    deepmd.train(train4,json_file,ckpt_file=ckpt)
