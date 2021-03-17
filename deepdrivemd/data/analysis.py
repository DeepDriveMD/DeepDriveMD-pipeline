from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor
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

    def apply_analysis_fn(
        self,
        fn: Callable,
        num_workers: Optional[int] = None,
        n: Optional[int] = None,
        data_file_suffix: str = ".h5",
        traj_file_suffix: str = ".dcd",
        structure_file_suffix: str = ".pdb",
    ) -> List[Any]:
        md_data = self.api.get_last_n_md_runs(
            n, data_file_suffix, traj_file_suffix, structure_file_suffix
        )
        output_data = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for data in tqdm(executor.map(fn, zip(md_data.values()))):
                output_data.append(data)
        return output_data
