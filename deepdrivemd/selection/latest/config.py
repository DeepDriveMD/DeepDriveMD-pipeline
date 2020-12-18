from deepdrivemd.config import ModelSelectionTaskConfig


class LatestCheckpointConfig(ModelSelectionTaskConfig):
    """Config for selecting the latest model checkpoint."""


if __name__ == "__main__":
    LatestCheckpointConfig().dump_yaml("latest_checkpoint_template.yaml")
