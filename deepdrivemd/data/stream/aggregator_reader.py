import adios2
import numpy as np
from deepdrivemd.utils import t1Dto2D
from pathlib import Path
from typing import List, Tuple
from deepdrivemd.data.stream.adios_utils import ADIOS_RW_FULL_API


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

        variables = {
            "step": (np.int32, 0),
            "rmsd": (np.float32, 0),
            "contact_map": (np.uint8, 1),
            "positions": (np.float32, 1),
            "velocities": (np.float32, 1),
            "md5": (str, 2),
        }

        connections = {0: (self.adios, self.io, self.stream)}

        ARW = ADIOS_RW_FULL_API(connections, variables)

        for i in range(N):
            status = ARW.read_step(0)
            if not status:
                break

            ARW.d_contact_map = np.unpackbits(ARW.d_contact_map)
            ARW.d_contact_map = t1Dto2D(ARW.d_contact_map)
            MD5s.append(ARW.d_md5)
            CMs.append(ARW.d_contact_map)
            STEPs.append(ARW.d_step[0])
            RMSDs.append(ARW.d_rmsd[0])
            POSITIONs.append(ARW.d_positions)
            VELOCITYs.append(ARW.d_velocities)

        return i, STEPs, MD5s, CMs, POSITIONs, VELOCITYs, RMSDs

    def next_cm(self, N: int) -> Tuple[int, List[np.ndarray]]:
        """
        Mini version of next_all where only contact maps are returned
        """
        CMs = []

        variables = {
            "contact_map": (np.uint8, 1),
        }

        connections = {0: (self.adios, self.io, self.stream)}
        ARW = ADIOS_RW_FULL_API(connections, variables)

        for i in range(N):
            status = ARW.read_step(0)
            if not status:
                break

            ARW.d_contact_map = np.unpackbits(ARW.d_contact_map)
            ARW.d_contact_map = t1Dto2D(ARW.d_contact_map)
            CMs.append(ARW.d_contact_map)

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
