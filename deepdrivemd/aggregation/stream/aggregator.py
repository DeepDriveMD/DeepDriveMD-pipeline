import itertools
import math
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import adios2
import numpy as np

from deepdrivemd.aggregation.stream.config import StreamAggregation
from deepdrivemd.data.stream.adios_utils import AdiosStreamStepRW
from deepdrivemd.data.stream.enumerations import DataStructure
from deepdrivemd.utils import Timer, intarray2hash, parse_args, timer


def find_input(cfg: StreamAggregation) -> List[str]:
    """Find adios streams to which simulations write.

    Parameters
    ----------
    cfg : StreamAggregation

    Returns
    -------
    List[str]
           a list of sst files associated with simulations
    """
    while True:
        bpfiles = list(map(str, list(cfg.experiment_directory.glob("*/*/*/md.bp*"))))
        if len(bpfiles) == cfg.n_sim:
            break
        print("In find_input: waiting for input")
        time.sleep(cfg.sleeptime_bpfiles)

    bpfiles.sort()
    return bpfiles


def connect_to_input(
    cfg: StreamAggregation, bpfiles: List[Path]
) -> Dict[int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]]:
    """Open adios streams for reading.

    Parameters
    ----------
    cfg : StreamAggregation
    bpfiles : List[Path]

    Returns
    -------
    Dict[int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]]
           key - simulation task id, value - tuple of the corresponding adios objects.

    """
    connections = {}
    bp_slice = math.ceil(cfg.n_sim / cfg.num_tasks)
    print("bp_slice = ", bp_slice)

    for i, bp in enumerate(bpfiles):
        bp = bp.replace(".sst", "")
        sim_dir = os.path.dirname(bp)
        task_md = os.path.basename(sim_dir)
        taskid_md = int(task_md.replace("task", ""))
        adios_md = sim_dir + "/adios.xml"

        print(f"taskid_md = {taskid_md}, i = {i}, {i*bp_slice}, {(i+1)*bp_slice}")

        if taskid_md // bp_slice == cfg.task_idx:
            adios = adios2.ADIOS(adios_md, True)
            io = adios.DeclareIO(task_md)
            io.SetParameters({"ControlModule": "epoll"})
            stream = io.Open(bp, adios2.Mode.Read)
            connections[taskid_md] = (adios, io, stream)

    return connections


def aggregate(
    cfg: StreamAggregation,
    connections: Dict[
        int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]
    ],
    aggregator_stream: adios2.adios2.Engine,
    aggregator_stream_4ml: adios2.adios2.Engine,
):
    """Read adios streams from a subset of simulations handled by this
    aggregator and write them to adios file to be used by machine learning and outlier search.

    Parameters
    ----------
    cfg : StreamAggregation
    connections : Dict[int, Tuple[adios2.adios2.ADIOS, adios2.adios2.IO, adios2.adios2.Engine]]
          key - task id, value - a tuple of adios objects
    aggregator_stream : adios2.adios2.Engine
          an adios stream of aggregated file to write to.

    Note
    ----
    If we do not need to save the data for postproduction, we can get rid of the aggregated
    adios file and replace it by SST stream.
    """

    variablesR = {
        "step": (np.int32, DataStructure.scalar),
        "positions": (np.float32, DataStructure.array),
        "velocities": (np.float32, DataStructure.array),
        "md5": (np.int64, DataStructure.array),
    }

    if cfg.model == "cvae":
        variablesR["contact_map"] = (np.uint8, DataStructure.array)
    elif cfg.model == "aae":
        variablesR["point_cloud"] = (np.float32, DataStructure.array)

    variablesW = {
        "step": (np.int32, DataStructure.scalar),
        "positions": (np.float32, DataStructure.array),
        "velocities": (np.float32, DataStructure.array),
        "md5": (str, DataStructure.scalar),
    }

    if cfg.model == "cvae":
        variablesW["contact_map"] = (np.uint8, DataStructure.array)
    elif cfg.model == "aae":
        variablesW["point_cloud"] = (np.float32, DataStructure.array)

    if cfg.model == "cvae":
        variablesW_4ml = {
            "contact_map": (np.uint8, DataStructure.array),
        }
    elif cfg.model == "aae":
        variablesW_4ml = {
            "point_cloud": (np.float32, DataStructure.array),
        }

    if cfg.compute_rmsd:
        variablesR["rmsd"] = (np.float32, DataStructure.scalar)
        variablesW["rmsd"] = (np.float32, DataStructure.scalar)

    if cfg.compute_zcentroid:
        variablesR["zcentroid"] = (np.float32, DataStructure.scalar)
        variablesW["zcentroid"] = (np.float32, DataStructure.scalar)

    if hasattr(cfg, "multi_ligand_table") and cfg.multi_ligand_table.is_file():
        variablesR["ligand"] = (np.int32, DataStructure.scalar)
        variablesR["natoms"] = (np.int32, DataStructure.scalar)
        variablesW["ligand"] = (np.int32, DataStructure.scalar)
        variablesW["natoms"] = (np.int32, DataStructure.scalar)

    print("In aggregate: variablesR = ", variablesR)
    print("In aggregate: variablesW_4ml = ", variablesW_4ml)
    print("In aggregate: variablesW = ", variablesW)
    sys.stdout.flush()

    ARW = AdiosStreamStepRW(connections, variablesR)

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
                """
                print("dir(ARW) = ", dir(ARW))
                sys.stdout.flush()
                print("contact_map.shape = ", ARW.d_contact_map.shape)
                print("contact_map.dtype = ", ARW.d_contact_map.dtype)
                print(ARW.d_contact_map)
                print("positions.shape = ", ARW.d_positions.shape)
                print("positions.dtype = ", ARW.d_positions.dtype)
                sys.stdout.flush()
                """

                ARW.write_step(aggregator_stream_4ml, variablesW_4ml, end_step=True)
                ARW.write_step(aggregator_stream, variablesW, end_step=False)
                aggregator_stream.write("dir", str(sim_task_id), end_step=True)
            else:
                print(f"NotReady in simulation {sim_task_id}")
                q.put(sim_task_id)
                continue

        timer("aggregator_iteration", -1)


if __name__ == "__main__":
    print(subprocess.getstatusoutput("hostname")[1])
    sys.stdout.flush()

    args = parse_args()
    cfg = StreamAggregation.from_yaml(args.config)

    print(cfg)

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

    bpaggregator_4ml = str(cfg.output_path / "agg_4ml.bp")
    aggregator_stream_4ml = adios2.open(
        name=bpaggregator_4ml,
        mode="w",
        config_file=str(cfg.adios_xml_agg_4ml),
        io_in_config_file="AggregatorOutput4ml",
    )

    aggregate(cfg, connections, aggregator_stream, aggregator_stream_4ml)

    # Currently there is an infinite loop in aggregate() and this statement should never be reached.
    aggregator_stream.close()
    aggregator_stream_4ml.close()
