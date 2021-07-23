import adios2
import numpy as np
from deepdrivemd.utils import t1Dto2D
from pathlib import Path  # noqa
from typing import List, Tuple  # noqa

"""
class StreamVariable:
    def __init__(self, io, stream, name: str, dtype: type, structure_type:int):

        self.var = io.InquireVariable(name)

        if(structure_type == 0): # scalar
            self._data = np.zeros(1, dtype=dtype)
            stream.Get(self.var, self._data)
        elif(structure_type == 1): # array
            shape = self.var.Shape()
            start = [0] * len(shape) # ndim
            self.var.SetSelection([start, shape])
            self._data = np.zeros(shape, dtype=dtype)
            stream.Get(self.var, self._data)
        else: # string
            self._data = stream.Get(self.var)

    @property
    def data(self) -> np.ndarray:
        return self._data
"""


class ADIOS_READER:
    """
    This class is used to read the next N steps from an adios file
    """

    def __init__(self, fn: str, config: Path, stream_name: str):
        """
        fn - adios file name, config - adios.xml, stream_name - name of the stream in adios.xml
        """
        self.adios = adios2.ADIOS(str(config), True)
        self.io = self.adios.DeclareIO(stream_name)
        self.stream = self.io.Open(fn, adios2.Mode.Read)

    def __del__(self):
        self.stream.Close()
        self.io.RemoveAllVariables()
        self.adios.RemoveAllIOs()

    def next_all(
        self, N: int
    ) -> Tuple[
        int,
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        Read the next N steps of all variables.
        Returns a tuple consisting of the number of actually read steps, and lists of numpy variables for each variable, each time step
        """
        CMs = []
        POSITIONs = []
        MD5s = []
        STEPs = []
        RMSDs = []
        VELOCITYs = []
        MD5s = []
        for i in range(N):
            status = self.stream.BeginStep(adios2.StepMode.Read, 0.0)
            if status != adios2.StepStatus.OK:
                break

            """
            step = StreamVariable(self.io, self.stream, "step", np.int32, 0)
            rmsd = StreamVariable(self.io, self.stream, "rmsd", np.float32, 0)
            cm = StreamVariable(self.io, self.stream, "contact_map", np.uint8, 1)
            positions = StreamVariable(self.io, self.stream, "positions", np.float32, 1)
            velocities = StreamVariable(self.io, self.stream, "velocities", np.float32, 1)
            md5 = StreamVariable(self.io, self.stream, "md5", str, 2)
            """

            step = np.zeros(1, dtype=np.int32)
            varStep = self.io.InquireVariable("step")
            self.stream.Get(varStep, step)

            rmsd = np.zeros(1, dtype=np.float32)
            varRMSD = self.io.InquireVariable("rmsd")
            self.stream.Get(varRMSD, rmsd)

            varCM = self.io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0] * ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.uint8)
            self.stream.Get(varCM, cm)

            varPositions = self.io.InquireVariable("positions")
            shapePositions = varPositions.Shape()
            ndimPositions = len(shapePositions)
            start = [0] * ndimPositions
            count = shapePositions
            varPositions.SetSelection([start, count])
            positions = np.zeros(shapePositions, dtype=np.float32)
            self.stream.Get(varPositions, positions)

            varVelocities = self.io.InquireVariable("velocities")
            shapeVelocities = varVelocities.Shape()
            ndimVelocities = len(shapeVelocities)
            start = [0] * ndimVelocities
            count = shapeVelocities
            varVelocities.SetSelection([start, count])
            velocities = np.zeros(shapeVelocities, dtype=np.float32)
            self.stream.Get(varVelocities, velocities)

            varMD5 = self.io.InquireVariable("md5")
            shapeMD5 = varMD5.Shape()
            ndimMD5 = len(shapeMD5)
            start = [0] * ndimMD5
            count = shapeMD5
            md5 = self.stream.Get(varMD5)

            self.stream.EndStep()

            """
            cm = np.unpackbits(cm.data)
            cm = t1Dto2D(cm)

            MD5s.append(md5.data)
            CMs.append(cm)
            STEPs.append(step.data[0])
            RMSDs.append(rmsd.data[0])
            POSITIONs.append(positions.data)
            VELOCITYs.append(velocities.data)
            """

            cm = np.unpackbits(cm)
            cm = t1Dto2D(cm)
            MD5s.append(md5)
            CMs.append(cm)
            STEPs.append(step[0])
            RMSDs.append(rmsd[0])
            POSITIONs.append(positions)
            VELOCITYs.append(velocities)

        return i, STEPs, MD5s, CMs, POSITIONs, VELOCITYs, RMSDs

    def next_cm(self, N: int) -> Tuple[int, List[np.ndarray]]:
        """
        Mini version of next_all where only contact maps are returned
        """
        CMs = []
        for i in range(N):
            status = self.stream.BeginStep(adios2.StepMode.Read, 0.0)
            if status != adios2.StepStatus.OK:
                break

            varCM = self.io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0] * ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.uint8)
            self.stream.Get(varCM, cm)

            self.stream.EndStep()

            cm = np.unpackbits(cm)
            cm = t1Dto2D(cm)
            CMs.append(cm)

        return i, CMs


class STREAMS:
    """
    The class gets lastN time steps from each aggregator
    """

    def __init__(
        self,
        file_list: List,
        config: str = "../aggregate/adios.xml",
        stream_name: str = "AdiosOutput",
        lastN: int = 2000,
        batch: int = 10000,
    ):
        """
        file_list - adios files from each aggregator,
        config - adios xml file for the files,
        stream_name - corresponding stream name in adios.xml,
        lastN - number of last steps to take from each adios file,
        batch - up to how many steps to read from each adios file at a time (call of next())
        """
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

    def next(
        self,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        Return a tuple of lists.
        Lists: contact maps, positions, md5s, steps, velocities, rmsds
        Each element of a list is a numpy array of the variable for the corresponding time step
        """
        cm = []
        positions = []
        md5 = []
        steps = []
        rmsds = []
        velocities = []
        lastN = self.lastN
        batch = self.batch
        for fn in self.readers:
            i, STEPs, MD5s, CMs, POSITIONs, VELOCITYs, RMSDs = self.readers[
                fn
            ].next_all(batch)
            if i >= lastN:
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
        z = list(
            map(
                lambda x: np.concatenate(x),
                (cm, positions, md5, steps, velocities, rmsds),
            )
        )
        return z[0], z[1], z[2], z[4], z[5]

    def next_cm(self) -> List[np.ndarray]:
        """
        Mini version of next(): only contact maps are returned
        """
        cm = []
        lastN = self.lastN
        batch = self.batch
        for fn in self.readers:
            i, CMs = self.readers[fn].next_cm(batch)
            if i >= lastN:
                self.cm[fn] = CMs[-lastN:]
            else:
                remain = lastN - i
                self.cm[fn] = self.cm[fn][-remain:] + CMs
            cm.append(self.cm[fn])
        z = np.concatenate(cm)
        return z
