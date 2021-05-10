import argparse
from pathlib import Path
from typing import Optional, Union, Tuple
from deepdrivemd.utils import Timer
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.selection.latest.config import LatestCheckpointConfig

PathLike = Union[str, Path]


def get_model_path(
    stage_idx: int = -1,
    task_idx: int = 0,
    api: Optional[DeepDriveMD_API] = None,
    experiment_dir: Optional[PathLike] = None,
) -> Optional[Tuple[Path, Path]]:
    r"""Get the current best model.

    Should be imported by other stages to retrieve the best model path.

    Parameters
    ----------
    api : DeepDriveMD_API, optional
        API to DeepDriveMD to access the machine learning model path.
    experiment_dir : Union[str, Path], optional
        Experiment directory to initialize DeepDriveMD_API.

    Returns
    -------
    None
        If model selection has not run before.
    model_config : Path, optional
        Path to the most recent model YAML configuration file
        selected by the model selection stage. Contains hyperparameters.
    model_checkpoint : Path, optional
        Path to the most recent model weights selected by the model
        selection stage.


    Raises
    ------
    ValueError
        If both `api` and `experiment_dir` are None.
    """
    if api is None and experiment_dir is None:
        raise ValueError("Both `api` and `experiment_dir` are None")

    if api is None:
        assert experiment_dir is not None
        api = DeepDriveMD_API(experiment_dir)

    data = api.model_selection_stage.read_task_json(stage_idx, task_idx)
    if data is None:
        return

    model_config = Path(data[0]["model_config"])
    model_checkpoint = Path(data[0]["model_checkpoint"])

    return model_config, model_checkpoint


def latest_checkpoint(
    api: DeepDriveMD_API,
    checkpoint_dir: str = "checkpoint",
    checkpoint_suffix: str = ".pt",
) -> Path:
    r"""Select latest PyTorch model checkpoint.

    Assuming the model outputs a `checkpoint_dir` directory with
    `checkpoint_suffix` checkpoint files with the form
    XXX_<epoch-index>_YYY_ZZZ...<`checkpoint_suffix`>,
    return the path to the latest training epoch model checkpoint.

    Parameters
    ----------
    api : DeepDriveMD_API
        API to DeepDriveMD to access the machine learning model path.
    checkpoint_dir : str, optional
        Name of the checkpoint directory inside the model path. Note,
        if checkpoint files are stored in the top level directory, set
        checkpoint_dir="".
    checkpoint_suffix : str, optional
        The file extension for checkpoint files (.pt, .h5, etc).

    Returns
    -------
    Path
        Path to the latest model checkpoint file.
    """
    task_dir = api.machine_learning_stage.task_dir()
    assert task_dir is not None
    checkpoint_files = task_dir.joinpath(checkpoint_dir).glob(f"*{checkpoint_suffix}")
    # Format: epoch-1-20200922-131947.pt, select latest epoch checkpoint
    return max(checkpoint_files, key=lambda x: int(x.name.split("-")[1]))


def latest_model_checkpoint(cfg: LatestCheckpointConfig):
    r"""Select the latest model checkpoint and write path to JSON.

    Find the latest model checkpoint written by the machine learning
    stage and write the path into a JSON file to be consumed by the
    agent stage.

    Parameters
    ----------
    cfg : LatestCheckpointConfig
        pydantic YAML configuration for model selection task.
    """
    api = DeepDriveMD_API(cfg.experiment_directory)

    # Check if there is a new model
    if cfg.stage_idx % cfg.retrain_freq == 0:
        # Select latest model checkpoint.
        model_checkpoint = latest_checkpoint(
            api, cfg.checkpoint_dir, cfg.checkpoint_suffix
        )
        # Get latest model YAML configuration.
        model_config = api.machine_learning_stage.config_path(
            cfg.stage_idx, cfg.task_idx
        )
    else:  # Use old model
        token = get_model_path(cfg.stage_idx - 1, cfg.task_idx, api)
        assert token is not None, f"{cfg.stage_idx - 1}, {cfg.task_idx}"
        model_config, model_checkpoint = token

    # Format data into JSON serializable list of dictionaries
    data = [
        {"model_checkpoint": str(model_checkpoint), "model_config": str(model_config)}
    ]
    # Dump metadata to disk for MD stage
    api.model_selection_stage.write_task_json(data, cfg.stage_idx, cfg.task_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    with Timer("model_selection_stage"):
        args = parse_args()
        cfg = LatestCheckpointConfig.from_yaml(args.config)
        latest_model_checkpoint(cfg)
