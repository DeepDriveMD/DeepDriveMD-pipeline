import deepmd
import os
import sys
from pathlib import Path

cwd = os.getcwd()
data_path = Path(sys.argv[1])
train = Path(sys.argv[2])
print("Begin training: "+str(train))
json_file = Path(train,"input.json")
ckpt = Path("model.ckpt")

if not train.exists():
    os.makedirs(train,exist_ok=True)
    deepmd.gen_input(data_path,json_file)
    deepmd.train(train,json_file)
else:
    deepmd.gen_input(data_path,json_file)
    deepmd.train(train,json_file,ckpt_file=ckpt)

print("Done  training: "+str(train))
