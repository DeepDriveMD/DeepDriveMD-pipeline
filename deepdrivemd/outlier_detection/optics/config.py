from deepdrivemd.config import ODBaseConfig


class OPTICSConfig(ODBaseConfig):
    """OPTICS outlier detection algorithm configuration."""

    # Number of outliers to detect (should be number of MD jobs + 1, incase errors)
    num_outliers: int = 100
    # Number of points in point cloud AAE
    num_points: int = 304
    # Inference batch size for encoder forward pass
    inference_batch_size: int = 128


if __name__ == "__main__":
    OPTICSConfig().dump_yaml("optics_template.yaml")
