from deepdrivemd.config import ModelSelectionTaskConfig


class LatestCheckpointConfig(ModelSelectionTaskConfig):
    """Config for selecting the latest model checkpoint."""

    # Number of DDMD iterations between training
    retrain_freq: int = 1


if __name__ == "__main__":
    LatestCheckpointConfig().dump_yaml("latest_checkpoint_template.yaml")
