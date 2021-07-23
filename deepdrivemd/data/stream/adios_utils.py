import adios2
import numpy as np
from typing import Dict, Tuple


class ADIOS_RW_FULL_API:
    """
    Read/Write step by step adios stream using Full API
    """

    def __init__(
        self,
        connections: Dict[
            int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]
        ],
        variables: Dict[str, Tuple[type, int]],
    ):
        """
        connections is a dictionary of adios connections; when reading adios SST network streams in aggregator, key is task id;
        when reading BP file in machine learning or outlier search, key is 0.
        variables - a dictionary of variables to read/write to/from adios stream; key - variable name in the stream, value - a tuple of data type and structure type (0 - scalar, 1 - np.array, 2 - string)
        """
        self.connections = connections
        self.variables = variables

    def read_step(self, sim_task_id: int) -> bool:
        """
        Read the next step from adios stream given by self.connections[sim_task_id]
        Adios variables are named "var_" + variable name in adios stream and stored in class.
        Data variables that hold the results are named "d_" + variable name in adios stream and stored in class.
        The variables are created on the fly with setattr from the keys of self.variables.
        If BeginStep() reports OK status, read data into self.d_-variable and return True. Otherwise, return False.
        """
        adios, io, stream = self.connections[sim_task_id]

        status = stream.BeginStep(adios2.StepMode.Read, 0.0)

        if not (status == adios2.StepStatus.OK):
            return False

        for v in self.variables:
            vname = "var_" + v
            dname = "d_" + v
            dtype = self.variables[v][0]
            structure_type = self.variables[v][1]
            setattr(self, vname, io.InquireVariable(v))
            if structure_type == 0:  # scalar
                setattr(self, dname, np.zeros(1, dtype=dtype))
                stream.Get(getattr(self, vname), getattr(self, dname))
            elif structure_type == 1:  # np.array
                shape = getattr(self, vname).Shape()
                start = [0] * len(shape)  # ndim
                getattr(self, vname).SetSelection([start, shape])
                setattr(self, dname, np.zeros(shape, dtype=dtype))
                stream.Get(getattr(self, vname), getattr(self, dname))
            else:  # string
                setattr(self, dname, stream.Get(getattr(self, vname)))
        stream.EndStep()
        return True

    def write_step(
        self, wstream, variables: Dict[str, Tuple[type, int]], end_step: bool = False
    ):
        """
        Write the next step from class "d_" variables into wstream adios stream.
        A different variable dictionary is given because, for example, md5 might change its datatype from
        np.array to a string.
        If end_step=True, set end_step=True for the last write, otherwise - set it to False.
        """
        for v in variables:
            dname = "d_" + v
            structure_type = variables[v][1]
            data = getattr(self, dname)
            end = False
            if end_step and v == variables[-1]:
                end = True

            if structure_type == 0:  # scalar
                wstream.write(v, data, end_step=end)
            elif structure_type == 1:  # np.array
                wstream.write(
                    v,
                    data,
                    list(data.shape),
                    [0] * len(data.shape),
                    list(data.shape),
                    end_step=end,
                )
