import n2p2
import os
import sys
from pathlib import Path

cwd = os.getcwd()
data_path = Path(sys.argv[1])/"input.data"
train = Path(sys.argv[2])
print("Begin training: "+str(train))

if not train.exists():
    n2p2.create_directory(train,data_path)
os.chdir(train)
n2p2.run_scaling()
n2p2.run_training()
n2p2.select_best_model()

print("Done  training: "+str(train))
