'''
====> Epoch: 0 Train:   Avg Disc loss: 8.0352   Avg AE loss: 7399.7970  Time: 1.7543
====> Epoch: 0 Valid:   Avg recon loss: 15285.2761      Time: 0.4237
====> Epoch: 1 Train:   Avg Disc loss: 3.0858   Avg AE loss: 7394.5719  Time: 1.6086
====> Epoch: 1 Valid:   Avg recon loss: 15285.7327      Time: 0.4295
'''

import sys
import re
import pandas as pd
import os

iterations = []
trains = []
epochs = []
avg_disc_losses = []
avg_ae_losses = []
avg_recon_losses = []
timesT = []
timesV = []


pattern1 = '====> Epoch:\s+(\d+)\s+Train:\s+Avg Disc loss:\s+(\-*\d+\.*\d+)\s+Avg AE loss:\s+(\-*\d+\.*\d+)\s+Time:\s+(\d+\.*\d+)'
pattern2 = '====> Epoch:\s+(\d+)\s+Valid:\s+Avg recon loss:\s+(\-*\d+\.*\d+)\s+Time:\s+(\d+\.*\d+)'

p1 = re.compile(pattern1)
p2 = re.compile(pattern2)

#r1 = re.findall(p1, "====> Epoch: 0 Train:   Avg Disc loss: 8.0352   Avg AE loss: 7399.7970  Time: 1.7543")
#print(r1)
#r2 = re.findall(p2, "====> Epoch: 0 Valid:   Avg recon loss: 15285.2761      Time: 0.4237")
#print(r2)

iteration = -1

fn = sys.argv[1]

dir = sys.argv[2]

with open(fn) as f:
    lines = f.readlines()
    lines = filter(lambda x: x.find("====> Epoch:") == 0, lines)

    even = True
    for line in lines:
        if(even):
            r = re.findall(p1, line)
            if(len(r) == 0):
                print(line)
                sys.exit(0)
            epoch = int(r[0][0])
            if(epoch == 0):
                iteration += 1
            epochs.append(epoch)
            iterations.append(iteration)
            avg_disc_loss = float(r[0][1])
            avg_disc_losses.append(avg_disc_loss)
            avg_ae_loss = float(r[0][2])
            avg_ae_losses.append(avg_ae_loss)
            timeT = float(r[0][3])
            timesT.append(timeT)
        else:
            r = re.findall(p2, line)
            epoch1 = int(r[0][0])
            if(epoch1 != epoch):
                print("Epochs: ", epoch, epoch1)
            avg_recon_loss = float(r[0][1])
            avg_recon_losses.append(avg_recon_loss)
            timeV = float(r[0][2])
            timesV.append(timeV)
        even = not even

df = pd.DataFrame({'iteration':iterations, 'epoch':epochs, 'avg_disc_loss': avg_disc_losses, 
                   'avg_ae_loss': avg_ae_losses, 'avg_recon_loss': avg_recon_losses, 
                   'timeT': timesT, 'timeV': timesV})

odir = '/p/gpfs1/yakushin/Outputs/' + dir + '/postproduction'
try:
    os.mkdir(odir)
except:
    pass

ofn = odir + '/losses.csv'

df.to_csv(ofn)
