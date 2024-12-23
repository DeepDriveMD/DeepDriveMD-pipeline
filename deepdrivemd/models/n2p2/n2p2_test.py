import n2p2
import os
from pathlib import Path

cwd = os.getcwd()
data_path = Path(cwd,"../../sim/nwchem/test_dir/input.data")
top = Path(cwd)
#scaling = Path(cwd,"scaling")
train1  = Path(cwd,"train-1")
train2  = Path(cwd,"train-2")
train3  = Path(cwd,"train-3")
train4  = Path(cwd,"train-4")

#n2p2.create_directories(data_path)

#os.chdir(scaling)
#n2p2.run_scaling()
#os.chdir(top)

n2p2.create_directory(train1,data_path)
os.chdir(train1)
n2p2.run_scaling()
n2p2.run_training()
n2p2.select_best_model()
os.chdir(top)

n2p2.create_directory(train2,data_path)
os.chdir(train2)
n2p2.run_scaling()
n2p2.run_training()
n2p2.select_best_model()
os.chdir(top)

n2p2.create_directory(train3,data_path)
os.chdir(train3)
n2p2.run_scaling()
n2p2.run_training()
n2p2.select_best_model()
os.chdir(top)

n2p2.create_directory(train4,data_path)
os.chdir(train4)
n2p2.run_scaling()
n2p2.run_training()
n2p2.select_best_model()
os.chdir(top)
