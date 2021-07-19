from deepdrivemd.config import AggregationTaskConfig
from pathlib import Path

class StreamAggregation(AggregationTaskConfig):
    # is rmsd used
    rmsd: bool = True
    # number of simulations
    n_sim: int = 12
    # if adios streams from simulations are not available, sleep for this number of seconds before trying to find them again
    sleeptime_bpfiles: int = 30
    # number of aggregators
    num_tasks: int = 2
    # path to adios xml configuration file for aggregator
    adios_xml_agg: Path = ''

if __name__ == "__main__":
    StreamAggregation().dump_yaml("stream_aggregation_template.yaml")
