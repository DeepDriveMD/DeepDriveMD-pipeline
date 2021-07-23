import numpy as np
from deepdrivemd.aggregation.stream.config import StreamAggregation

# from deepdrivemd.misc.aggregator_reader import StreamVariable
from deepdrivemd.utils import Timer, timer, parse_args, intarray2hash
import time
import os
import sys
import adios2
import math
import queue
import subprocess
import itertools


def find_input(cfg: StreamAggregation):
    """
    Find and return simulation adios streams.
    Wait until those are created by simulations.
    Returns the list of *.sst files associated with the corresponding adios streams.
    """
    while True:
        bpfiles = list(map(str, list(cfg.experiment_directory.glob("*/*/*/md.bp*"))))
        if len(bpfiles) == cfg.n_sim:
            break
        print("In find_input: waiting for input")
        time.sleep(cfg.sleeptime_bpfiles)

    bpfiles.sort()
    return bpfiles


def connect_to_input(cfg: StreamAggregation, bpfiles):
    """
    Open adios streams for reading and return a dictionary: key - simulation task id,
    value - tuple of the corresponding adios objects
    """
    connections = {}
    bp_slice = math.ceil(cfg.n_sim / cfg.num_tasks)
    print("bp_slice = ", bp_slice)
    i = 0
    for bp in bpfiles:
        bp = bp.replace(".sst", "")
        dir = os.path.dirname(bp)
        task_md = os.path.basename(dir)
        taskid_md = int(task_md.replace("task", ""))
        adios_md = dir + "/adios.xml"

        taskid_agg = cfg.task_idx

        print(f"taskid_md = {taskid_md}, i = {i}, {i*bp_slice}, {(i+1)*bp_slice}")

        if taskid_md // bp_slice == taskid_agg:
            adios = adios2.ADIOS(adios_md, True)
            io = adios.DeclareIO(task_md)
            io.SetParameters({"ControlModule": "epoll"})
            stream = io.Open(bp, adios2.Mode.Read)
            connections[taskid_md] = (adios, io, stream)

        i += 1

    return connections


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


def aggregate(cfg: StreamAggregation, connections, aggregator_stream):
    """
    Read adios streams from a subset of simulations handled by this aggregator and write them to adios file to be used
    by machine learning and outlier search. If we do not need to save the data for postproduction, we can get rid of the aggregated
    adios file and replace it by SST stream.
    """

    variablesR = {
        "step": (np.int32, 0),
        "rmsd": (np.float32, 0),
        "contact_map": (np.uint8, 1),
        "positions": (np.float32, 1),
        "velocities": (np.float32, 1),
        "md5": (np.int64, 1),
    }

    variablesW = {
        "step": (np.int32, 0),
        "rmsd": (np.float32, 0),
        "contact_map": (np.uint8, 1),
        "positions": (np.float32, 1),
        "velocities": (np.float32, 1),
        "md5": (str, 0),
    }

    ARW = ADIOS_RW_FULL_API(connections, variablesR)

    # infinite loop over simulation reporting steps
    for iteration in itertools.count(0):
        timer("aggregator_iteration", 1)
        print("iteration = ", iteration)

        q = queue.Queue()
        for s in connections.keys():
            q.put(s)

        # Read data from each simulation and write it to the aggregated adios file
        # If the data is not ready yet, go to the next simulation and revisit the current one later
        while not q.empty():
            sim_task_id = q.get()

            status = ARW.read_step(sim_task_id)
            if status:
                ARW.d_md5 = intarray2hash(ARW.d_md5)
                ARW.write_step(aggregator_stream, variablesW, end_step=False)
                aggregator_stream.write("dir", str(sim_task_id), end_step=True)
            else:
                print(f"NotReady in simulation {sim_task_id}")
                q.put(sim_task_id)
                continue

            """
            adios, io, stream = connections[sim_task_id]

            status = stream.BeginStep(adios2.StepMode.Read, 0.0)

            if status == adios2.StepStatus.NotReady:
                print(f"NotReady in simulation {sim_task_id}")
                q.put(sim_task_id)
                continue
            if status == adios2.StepStatus.EndOfStream:
                print(f"EndOfStream in simulation {sim_task_id}")
                q.put(sim_task_id)
                continue
            if status == adios2.StepStatus.OtherError:
                print(f"OtherError in simulation {sim_task_id}")
                q.put(sim_task_id)
                continue

            step = np.zeros(1, dtype=np.int32)
            varStep = io.InquireVariable("step")
            stream.Get(varStep, step)

            rmsd = np.zeros(1, dtype=np.float32)
            varRMSD = io.InquireVariable("rmsd")
            stream.Get(varRMSD, rmsd)

            varCM = io.InquireVariable("contact_map")
            shapeCM = varCM.Shape()
            ndimCM = len(shapeCM)
            start = [0] * ndimCM
            count = shapeCM
            varCM.SetSelection([start, count])
            cm = np.zeros(shapeCM, dtype=np.uint8)
            stream.Get(varCM, cm)

            varPositions = io.InquireVariable("positions")
            shapePositions = varPositions.Shape()
            ndimPositions = len(shapePositions)
            start = [0] * ndimPositions
            count = shapePositions
            varPositions.SetSelection([start, count])
            positions = np.zeros(shapePositions, dtype=np.float32)
            stream.Get(varPositions, positions)

            varVelocities = io.InquireVariable("velocities")
            shapeVelocities = varVelocities.Shape()
            ndimVelocities = len(shapeVelocities)
            start = [0] * ndimVelocities
            count = shapeVelocities
            varVelocities.SetSelection([start, count])
            velocities = np.zeros(shapeVelocities, dtype=np.float32)
            stream.Get(varVelocities, velocities)

            varMD5 = io.InquireVariable("md5")
            shapeMD5 = varMD5.Shape()
            ndimMD5 = len(shapeMD5)
            start = [0] * ndimMD5
            count = shapeMD5
            varMD5.SetSelection([start, count])
            md5 = np.zeros(shapeMD5, dtype=np.int64)
            stream.Get(varMD5, md5)

            stream.EndStep()

            md5 = intarray2hash(md5)

            aggregator_stream.write("md5", md5)
            aggregator_stream.write("step", step)
            aggregator_stream.write("rmsd", rmsd)
            aggregator_stream.write("dir", str(sim_task_id))
            aggregator_stream.write(
                "positions",
                positions,
                list(positions.shape),
                [0] * len(positions.shape),
                list(positions.shape),
            )
            aggregator_stream.write(
                "velocities",
                velocities,
                list(velocities.shape),
                [0] * len(velocities.shape),
                list(velocities.shape),
            )
            aggregator_stream.write(
                "contact_map",
                cm,
                list(cm.shape),
                [0] * len(cm.shape),
                list(cm.shape),
                end_step=True,
            )
            """

        timer("aggregator_iteration", -1)


if __name__ == "__main__":
    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()

    args = parse_args()
    cfg = StreamAggregation.from_yaml(args.config)

    with Timer("aggregator_find_adios_files"):
        bpfiles = find_input(cfg)
    print("bpfiles = ", bpfiles)
    print("len(bpfiles) = ", len(bpfiles))

    with Timer("aggregator_connect"):
        connections = connect_to_input(cfg, bpfiles)
    print("connections = ", connections)
    print("len(connections) = ", len(connections))

    bpaggregator = str(cfg.output_path / "agg.bp")

    aggregator_stream = adios2.open(
        name=bpaggregator,
        mode="w",
        config_file=str(cfg.adios_xml_agg),
        io_in_config_file="AggregatorOutput",
    )

    aggregate(cfg, connections, aggregator_stream)

    # Currently there is an infinite loop in aggregate() and this statement should never be reached.
    aggregator_stream.close()
