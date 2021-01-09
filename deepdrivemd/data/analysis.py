from pathlib import Path
from typing import Union, List, Dict, Any
from deepdrivemd.data.api import DeepDriveMD_API

PathLike = Union[str, Path]


class DeepDriveMD_Analysis:
    def __init__(self, experiment_directory: PathLike):
        self.api = DeepDriveMD_API(experiment_directory)

    def get_agent_data(self, iterations: int = -1) -> List[List[Dict[str, Any]]]:
        if iterations == -1:
            iterations = self.api.get_total_iterations()
        agent_json_data = [
            self.api.agent_stage.read_task_json(stage_idx)
            for stage_idx in range(iterations)
        ]
        assert None not in agent_json_data
        return agent_json_data
