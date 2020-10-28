import itertools
from pathlib import Path
from typing import List


class DeepDriveMD_API:
    def __init__(self, experiment_directory: Path):
        self.experiment_dir = Path(experiment_directory)

    def get_last_n_hdf5(self, n=None) -> List[str]:
        # Run dirs: f"run_{deepdrivemd_iteration:03d}_{sim_task_id:04d}"
        run_dirs = self.experiment_dir.joinpath("md_runs").glob("*")
        # Remove any potential files
        run_dirs = filter(lambda x: x.is_dir(), run_dirs)
        # Convert pathlib Path to str
        run_dirs = map(lambda x: x.as_posix(), run_dirs)
        # Sort by deepdrivemd iteration and sim task id (deepdrivemd_iteration, sim_task_id)
        run_dirs = sorted(run_dirs, key=lambda x: tuple(x.split("_")[1:]))
        # Reverse sort to get last n
        run_dirs = reversed(run_dirs)
        # Evaluate generator up to n items
        return list(itertools.islice(run_dirs, n))
