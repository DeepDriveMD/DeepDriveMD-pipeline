from pathlib import Path
import numpy as np
from typing import Union, List, Dict, Any
from deepdrivemd.data.api import DeepDriveMD_API
from deepdrivemd.data.utils import parse_h5

PathLike = Union[str, Path]


class DeepDriveMD_Analysis:
    def __init__(self, experiment_directory: PathLike):
        self.api = DeepDriveMD_API(experiment_directory)

    def get_agent_json(self, iterations: int = -1) -> List[List[Dict[str, Any]]]:
        if iterations == -1:
            iterations = self.api.get_total_iterations()
        agent_json_data = [
            self.api.agent_stage.read_task_json(stage_idx)
            for stage_idx in range(iterations)
        ]
        assert None not in agent_json_data
        return agent_json_data

    def get_agent_h5(
        self, iterations: int = -1, fields: List[str] = []
    ) -> List[Dict[str, np.ndarray]]:
        if iterations == -1:
            iterations = self.api.get_total_iterations()
        h5_data = [
            parse_h5(
                next(self.api.agent_stage.stage_dir(stage_idx).glob("**/*.h5")), fields
            )
            for stage_idx in range(iterations)
        ]
        return h5_data
