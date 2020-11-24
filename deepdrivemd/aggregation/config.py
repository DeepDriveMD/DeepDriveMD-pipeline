from deepdrivemd.config import AggregationBaseConfig


class BasicAggegation(AggregationBaseConfig):
    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True
