import adios2
import numpy as np


class ADIOS_RW_FULL_API:
    def __init__(self, connections, variables):
        self.connections = connections
        self.variables = variables

    def read_step(self, sim_task_id):
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

    def write_step(self, wstream, variables, end_step=False):
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
