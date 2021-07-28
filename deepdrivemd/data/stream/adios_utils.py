import adios2
import numpy as np
from typing import Dict, Tuple


class ADIOS_RW_FULL_API:
    """Read/Write step by step adios stream using Full API

    Attributes
    ----------
    connections : Dict[int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]]
         dictionary of adios connections; key - integer, in aggregator it is simulation task id;
         value - a tuple of adios objects
    variables : Dict[str, Tuple[type, int]],
         dictionary describing variables; key - adios column name, value - a tuple of variable type and
         enumeration describing the structure type: 0 - scalar, 1 - numpy array, 2 - string;
         other class attributes are created on the fly using `setattr`: for each key two attributes
         are created: `var_<key>` - adios variable, `d_<key>` - data which stores the result of reading
         a particular variable `key` from a step of adios stream.
    """

    def __init__(
        self,
        connections: Dict[
            int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]
        ],
        variables: Dict[str, Tuple[type, int]],
    ):
        """Constructor.

        Parameters
        ----------
        connections : Dict[int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]]
             dictionary of adios connections; key - integer, in aggregator it is simulation task id,
             value - a tuple of adios objects
        variables : Dict[str, Tuple[type, int]]
             dictionary describing variables; key - adios column name, value - a tuple of variable type and
             enumeration describing the structure type: 0 - scalar, 1 - numpy array, 2 - string.
        """
        self.connections = connections
        self.variables = variables

    def read_step(self, sim_task_id: int) -> bool:
        """Read the next step from adios stream given by `connections[sim_task_id]`.

        Parameters
        ----------
        sim_task_id : int
             is used as a key to get the corresponding adios objects from `connections`

        Returns
        -------
        bool
             `True` if reading a step succeeded, `False` - otherwise.
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
        self,
        wstream: adios2.adios2.Engine,
        variables: Dict[str, Tuple[type, int]],
        end_step: bool = False,
    ):
        """Write the next step from class `d_...` variables into `wstream` adios stream.

        Parameters
        ----------
        wstream : adios2.adios2.Engine
             adios stream to which the data is written
        variables : Dict[str, Tuple[type, int]]
             a dictionary indexed by adios column names, value is a tuple - data type, structure type;
             structure type can be 0 - scalar, 1 - np.array, 2 - str
        end_step : bool, default = False
             if this is `True`, the write of the last variable would be marked by `end_step = True`
             meaning that the step writing is done; otherwise, terminating the step should be done
             outside of the method
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
