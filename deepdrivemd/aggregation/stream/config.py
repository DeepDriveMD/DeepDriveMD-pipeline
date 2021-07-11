from typing import Optional
from deepdrivemd.config import AggregationTaskConfig
from pathlib import Path

class StreamAggregation(AggregationTaskConfig):
    rmsd: bool = True
    contact_map: bool = False
    verbose: bool = True
    n_sim: int = 12
    sleeptime_bpfiles: int = 30
    num_tasks: int = 2
    adios_xml_sim: Path = ''
    adios_xml_agg: Path = ''

if __name__ == "__main__":
    StreamAggregation().dump_yaml("stream_aggregation_template.yaml")
