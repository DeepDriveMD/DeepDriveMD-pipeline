from deepdrivemd.config import ODBaseConfig


class LOFConfig(ODBaseConfig):
    """LOF outlier detection algorithm configuration."""

    # Number of outliers to detect (should be number of MD jobs + 1, incase errors)
    num_outliers: int = 100
    # Number of points in point cloud AAE
    num_points: int = 304
    # Inference batch size for encoder forward pass
    inference_batch_size: int = 128
    # Inference forward pass device
    device: str = "cuda:0"


if __name__ == "__main__":
    LOFConfig().dump_yaml("lof_template.yaml")
