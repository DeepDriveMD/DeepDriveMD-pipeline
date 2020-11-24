from deepdrivemd.config import AggregationBaseConfig


class BasicAggegation(AggregationBaseConfig):
    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True


if __name__ == "__main__":
    BasicAggegation().dump_yaml("basic_aggregation_template.yaml")
