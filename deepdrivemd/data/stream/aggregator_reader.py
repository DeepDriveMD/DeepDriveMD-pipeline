import adios2
import numpy as np

# from deepdrivemd.utils import t1Dto2D
from pathlib import Path
from typing import List, Dict, Union
from deepdrivemd.data.stream.adios_utils import AdiosStreamStepRW
from deepdrivemd.data.stream.enumerations import DataStructure


class StreamVariable:
    """This class is used to read a variable from BP file.

    Attributes
    ----------
    name : str
         variable name in adios file
    dtype : type
         variable type, for example, np.uint8
    structure : DataStructure
         enumeration: array, scalar, string
    total : List
         list of variable values for different steps
    """

    def __init__(self, name: str, dtype: type, structure: DataStructure):
        """
        Parameters
        ----------
        name : str
             variable name in adios file
        dtype : type
             variable type, for example, np.uint8
        structure : DataStructure
             structure type: array, scalar, string

        """
        self.name = name
        self.dtype = dtype
        self.structure = structure
        self.total = []

    def next(self, ARW: AdiosStreamStepRW):
        """Get the variable value for the next time step and append it to `total`.

        Parameters
        ----------
        ARW : AdiosStreamStepRW
             low level object for reading data from ADIOS stream (BP file or SST stream)

        """

        var = getattr(ARW, "d_" + self.name)
        self.total.append(var)


class StreamContactMapVariable(StreamVariable):
    """Implementation of `StreamVariable` that handles contact maps:
    unpack bits to 1D array, convert 1D array to 2D array.
    """

    def next(self, ARW):
        var = getattr(ARW, "d_" + self.name)
        # var = np.unpackbits(var)
        # var = t1Dto2D(var)
        self.total.append(var)


class StreamScalarVariable(StreamVariable):
    """Implementation of `StreamVariable` that handles scalar variables."""

    def next(self, ARW):
        var = getattr(ARW, "d_" + self.name)
        self.total.append(var[0])


class AdiosReader:
    """This class is used to read the next `N` steps from an adios stream.

    Attributes
    ----------
    adios : adios2.adios2.ADIOS
    io : adios2.adios2.IO
    stream : adios2.adios2.Engine
    """

    def __init__(
        self, fn: str, config: Path, stream_name: str, variables: List[StreamVariable]
    ):
        """
        Parameters
        ----------
        fn: str
            file name of bp file or sst socket (without sst extension)
        config : Path
            path to `adios.xml` file
        stream_name : str
            name of a stream in `adios.xml` file
        """

        print("config=", str(config))
        print("fn=", fn)
        print("stream_name=", stream_name)
        print("variables=", str(variables))
        import sys

        sys.stdout.flush()

        self.adios = adios2.ADIOS(str(config), True)
        self.io = self.adios.DeclareIO(stream_name)
        self.stream = self.io.Open(fn, adios2.Mode.Read)

        self.variables = variables
        self.connections = {0: (self.adios, self.io, self.stream)}

    def __del__(self):
        """Destructor: clean the adios resources"""
        self.stream.Close()
        self.io.RemoveAllVariables()
        self.adios.RemoveAllIOs()

    def next(self, N: int) -> Dict[str, Union[np.array, str, int, float]]:
        """Read the next `N` steps of all variables.

        Parameters
        ----------
        N : int
            read that many steps

        Returns
        -------
        Dict[str, Union[np.array, str, int, float]]
            values for different variables whose names are used as keys
        """

        vvv = {}
        for v in self.variables:
            vvv[v.name] = (v.dtype, v.structure)
            v.total = []

        ARW = AdiosStreamStepRW(self.connections, vvv)

        for i in range(N):
            status = ARW.read_step(0)
            if not status:
                break
            for v in self.variables:
                v.next(ARW)

        output = {"steps_read": i}
        for v in self.variables:
            output[v.name] = v.total.copy()

        return output


class Streams:
    """The class keeps `lastN` steps from each aggregator

    Attributes
    ----------
    readers : Dict[str, AdiosReader]
          a dictionary of `AdiosReader` indexed by the corresponding adios file name
    positions : Dict[str, np.ndarray]
    md5 : Dict[str, str]
    steps : Dict[str, np.ndarray]
    rmsds : Dict[str, np.ndarray]
    cm : Dict[str, np.ndarray]
    velocities : Dict[str, np.ndarray]
    lastN : int
         keep that many last steps from each aggregator
    batch : int
         up to how many steps to read from each adios file at a time

    """

    def __init__(
        self,
        files: List[str],
        variables: List[StreamVariable],
        config: Path = Path("../aggregate/adios.xml"),
        stream_name: str = "AdiosOutput",
        lastN: int = 2000,
        batch: int = 10000,
    ):
        """
        Parameters
        ----------
        files : List[str]
             adios files from each aggregator,
        variables: List[StreamVariable]
             list of variables to read from the aggegator file
        config : Path
             adios xml file for the files,
        stream_name : str
             corresponding stream name in adios.xml
        lastN : int
             number of last steps to keep from each adios file
        batch : int
             up to how many steps to read from each adios file at a time (call of `next()`)
        """

        self.variables = variables
        self.vnames = list(map(lambda x: x.name, variables))

        self.readers = {}

        for v in self.vnames:
            cname = "c_" + v
            setattr(self, cname, {})

        self.lastN = lastN
        self.batch = batch

        for fn in files:
            self.readers[fn] = AdiosReader(fn, config, stream_name, variables)
            for v in self.vnames:
                cname = "c_" + v
                cache = getattr(self, cname)
                cache[fn] = []

    def next(
        self,
    ) -> Dict[str, Union[np.array, int, float, str]]:
        """Provide `lastN` steps from each aggregator

        Returns
        -------
        Dict[str, Union[np.array, int, float, str]]
            values for the the variables whose names are used as keys
        """

        lastN = self.lastN
        batch = self.batch
        for fn in self.readers:
            nextbatch = self.readers[fn].next(batch)

            i = nextbatch["steps_read"]
            print(f"lastN = {lastN}, batch = {batch}, i = {i}")
            if i >= lastN:
                for j, v in enumerate(self.vnames):
                    cname = "c_" + v
                    cache = getattr(self, cname)
                    cache[fn] = nextbatch[v][-lastN:]
            else:
                remain = lastN - i
                for j, v in enumerate(self.vnames):
                    cname = "c_" + v
                    cache = getattr(self, cname)
                    cache[fn] = cache[fn][-remain:] + nextbatch[v]

        output = {}
        print(f"vnames = {self.vnames}")
        for v in self.vnames:
            cname = "c_" + v
            cache = getattr(self, cname)

            for k in cache:
                print("k=",k)
                print("v=", cache[k])
                print("len(v)=", len(cache[k]))

            values = list(cache.values())
            print("before filter: len(values) = ", len(values))
            values = list(filter(lambda x: len(x) > 0, values))
            print("after filter: len(values) = ", len(values))            
            import sys; sys.stdout.flush()
            output[v] = np.concatenate(values)

        return output
