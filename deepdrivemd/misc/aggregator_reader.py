import adios2
import numpy as np
import sys
from deepdrivemd.utils import t1Dto2D, t2Dto1D

class ADIOS_READER:
    def __init__(self, fn, config, stream_name):
        self.adios = adios2.ADIOS(config, True)
        self.io = self.adios.DeclareIO(stream_name)
        self.stream = self.io.Open(fn, adios2.Mode.Read)
    def __del__(self):
        self.stream.Close()
        self.io.RemoveAllVariables()
        self.adios.RemoveAllIOs()
    def next_all(self,N):
        CMs = []
        POSITIONs = []
        MD5s = []
        STEPs = []
        RMSDs = []
        VELOCITYs = []
        MD5s = []
        for i in range(N):
            status = self.stream.BeginStep(adios2.StepMode.Read, 0.0)
            if(status != adios2.StepStatus.OK):
                break

            stepA = np.zeros(1, dtype=np.int32)
            varStep = self.io.InquireVariable("step")
            self.stream.Get(varStep, stepA)

            rmsdA = np.zeros(1, dtype=np.float32)
            varRMSD = self.io.InquireVariable("rmsd")
            self.stream.Get(varRMSD, rmsdA)

            varCM = self.io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0]*ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.uint8)
            self.stream.Get(varCM, cm)

            varPositions = self.io.InquireVariable("positions")
            shapePositions = varPositions.Shape()
            ndimPositions = len(shapePositions)
            start = [0]*ndimPositions
            count = shapePositions
            varPositions.SetSelection([start, count])
            positions = np.zeros(shapePositions, dtype=np.float32)
            self.stream.Get(varPositions, positions)


            varVelocities = self.io.InquireVariable("velocities")
            shapeVelocities = varVelocities.Shape()
            ndimVelocities = len(shapeVelocities)
            start = [0]*ndimVelocities
            count = shapeVelocities
            varVelocities.SetSelection([start, count])
            velocities = np.zeros(shapeVelocities, dtype=np.float32)
            self.stream.Get(varVelocities, velocities)

            varMD5 = self.io.InquireVariable("md5")
            shapeMD5 = varMD5.Shape()
            ndimMD5 = len(shapeMD5)
            start = [0]*ndimMD5
            count = shapeMD5
            md5 = self.stream.Get(varMD5)

            self.stream.EndStep()

            step = stepA[0]
            rmsd = rmsdA[0]

            cm = np.unpackbits(cm)
            cm = t1Dto2D(cm)

            MD5s.append(md5)
            CMs.append(cm)
            STEPs.append(step)
            RMSDs.append(rmsd)
            POSITIONs.append(positions)
            VELOCITYs.append(velocities)

        return i, STEPs, MD5s, CMs, POSITIONs, VELOCITYs, RMSDs

    def next_cm(self,N):
        CMs = []
        for i in range(N):
            status = self.stream.BeginStep(adios2.StepMode.Read, 0.0)
            if(status != adios2.StepStatus.OK):
                break

            varCM = self.io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0]*ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.uint8)
            self.stream.Get(varCM, cm)

            self.stream.EndStep()

            cm = np.unpackbits(cm)
            cm = t1Dto2D(cm)
            CMs.append(cm)

        return i, CMs


def test1():
    for j in range(20):
        print('='*30)
        i, steps, md5s, cms, positions, velocities = ar.next_all(n)
        print(f"j={j}, i={i}")
        print(f"md5s = {md5s}")
        print(f"steps = {steps}")

    print(type(positions[0]))
    print(type(cms[0]))
    print(type(md5s[0]))
    print(type(steps[0]))
    print(type(velocities[0]))
    print(positions[0].shape)
    print(cms[0].shape)
    print(velocities[0].shape)

def test2():
    for j in range(7):
        print('*'*30)
        i, cms = ar.next_cm(n)
        print(f"j={j}, i={i}")

    print(type(cms[0]))
    print(cms[0].shape)



class STREAMS:
    def __init__(self, file_list, config="../aggregate/adios.xml", 
                 stream_name="AdiosOutput", lastN=2000, batch=10000):
        self.readers = {}
        self.positions = {}
        self.md5 = {}
        self.steps = {}
        self.rmsds = {}
        self.cm = {}
        self.velocities = {}
        self.lastN = lastN
        self.batch = batch
        for fn in file_list:
            self.readers[fn] = ADIOS_READER(fn, config, stream_name)
            self.positions[fn] = []
            self.md5[fn] = []
            self.steps[fn] = []
            self.rmsds[fn] = []
            self.cm[fn] = []
            self.velocities[fn] = []
    def next(self):
        cm = []
        positions = []
        md5 = []
        steps = []
        rmsds = []
        velocities = []
        lastN = self.lastN
        batch = self.batch
        for fn in self.readers:
            i, STEPs, MD5s, CMs, POSITIONs, VELOCITYs, RMSDs = self.readers[fn].next_all(batch)
            if(i >= lastN):
                self.positions[fn] = POSITIONs[-lastN:]
                self.cm[fn] = CMs[-lastN:]
                self.steps[fn] = STEPs[-lastN:]
                self.rmsds[fn] = RMSDs[-lastN:]
                self.md5[fn] = MD5s[-lastN:]
                self.velocities[fn] = VELOCITYs[-lastN:]
            else:
                remain = lastN - i
                self.positions[fn] = self.positions[fn][-remain:] + POSITIONs
                self.cm[fn] = self.cm[fn][-remain:] + CMs
                self.steps[fn] = self.steps[fn][-remain:] + STEPs
                self.rmsds[fn] = self.rmsds[fn][-remain:] + RMSDs
                self.md5[fn] = self.md5[fn][-remain:] + MD5s
                self.velocities[fn] = self.velocities[fn][-remain:] + VELOCITYs
            cm.append(self.cm[fn])
            positions.append(self.positions[fn])
            velocities.append(self.velocities[fn])
            md5.append(self.md5[fn])
            steps.append(self.steps[fn])
            rmsds.append(self.rmsds[fn])
        z = list(map(lambda x: np.concatenate(x), (cm, positions, md5, steps, velocities, rmsds)))
        return z[0], z[1], z[2], z[4], z[5]

    def next_cm(self):
        cm = []
        lastN = self.lastN
        batch = self.batch
        for fn in self.readers:
            i, CMs  = self.readers[fn].next_cm(batch)
            if(i >= lastN):
                self.cm[fn] = CMs[-lastN:]
            else:
                remain = lastN - i
                self.cm[fn] = self.cm[fn][-remain:] + CMs
            cm.append(self.cm[fn])
        z = np.concatenate(cm)
        return z


def test3():
    s = STREAMS(['aggregator0.bp'])
    z = s.next_cm()
    print(type(z))
    print(z.shape)


def test4():
    s = STREAMS(['aggregator0.bp','aggregator1.bp'])
    z = s.next()
    n = len(z)
    print(n)
    for i in range(n):
        print(i)
        print(type(z[i]))
        print(z[i].shape)



if(__name__ == '__main__'):

    fn = 'aggregator0.bp'
    ADIOS_XML_AGGREGATOR = 'adios.xml'
    stream_name = 'AggregatorOutput'

    ar = ADIOS_READER('aggregator0.bp', 'adios.xml', 'AggregatorOutput')
    n = 11

    test1()
    print("&"*30)
    test2()
    print("&"*30)
    test3()
    print("&"*30)
    test4()
