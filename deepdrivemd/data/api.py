import json
import itertools
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Callable
import MDAnalysis

PathLike = Union[str, Path]


def glob_file_from_dirs(dirs: List[str], pattern: str) -> List[str]:
    """Return a list of all items matching `pattern` in multiple `dirs`."""
    return [next(Path(d).glob(pattern)).as_posix() for d in dirs]


class Stage_API:
    @staticmethod
    def task_name(task_idx: int):
        return f"task{task_idx:04d}"

    @staticmethod
    def stage_name(stage_idx: int):
        return f"stage{stage_idx:04d}"

    @staticmethod
    def unique_name(task_path: Path) -> str:
        # <stage_name>_<task_name>
        return f"{task_path.parent.name}_{task_path.name}"

    @staticmethod
    def get_latest(
        path: Path, pattern: str, is_dir: bool = False, key: Callable = lambda x: x
    ) -> Optional[Path]:
        matches = list(filter(lambda p: p.is_dir() == is_dir, path.glob(pattern)))
        if not matches:
            return None
        return max(matches, key=key)

    @staticmethod
    def get_count(path: Path, pattern: str, is_dir: bool = False) -> int:
        matches = list(filter(lambda p: p.is_dir() == is_dir, path.glob(pattern)))
        return len(matches)

    def __init__(self, experiment_dir, stage_dir_name):
        self.experiment_dir = experiment_dir
        self._stage_dir_name = stage_dir_name

    @property
    def runs_dir(self) -> Path:
        return self.experiment_dir.joinpath(self._stage_dir_name)

    def stage_dir(self, stage_idx: int = -1) -> Optional[Path]:
        r"""Return the stage directory containing task subdirectories.

        Each stage type has a directory containing subdirectories stageXXXX.
        In each stageXXXX there are several task directories labeled taskXXXX.
        This function returns a particular stageXXXX directory selected with
        `stage_idx`. Each iteration of DeepDriveMD corresponds to a stageXXXX
        directory, they are labeled in increasing order.
        """
        if stage_idx == -1:
            return self.get_latest(self.runs_dir, pattern="stage*", is_dir=True)
        return self.runs_dir.joinpath(self.stage_name(stage_idx))

    def stage_dir_count(self) -> int:
        r"""Return the number of stage directories."""
        return self.get_count(self.runs_dir, pattern="stage*", is_dir=True)

    def task_dir(
        self, stage_idx: int = -1, task_idx: int = 0, mkdir: bool = False
    ) -> Optional[Path]:
        _stage_dir = self.stage_dir(stage_idx)
        if _stage_dir is None:
            return None
        _task_dir = _stage_dir.joinpath(self.task_name(task_idx))

        if mkdir:
            _task_dir.mkdir(exist_ok=True, parents=True)

        return _task_dir

    def _task_file_path(
        self, stage_idx: int = -1, task_idx: int = 0, suffix=".yaml"
    ) -> Optional[Path]:
        _task_dir = self.task_dir(stage_idx, task_idx)
        if _task_dir is None:
            return None
        file_name = f"{self.unique_name(_task_dir)}{suffix}"
        return _task_dir.joinpath(file_name)

    def config_path(self, stage_idx: int = -1, task_idx: int = 0) -> Optional[Path]:
        return self._task_file_path(stage_idx, task_idx, suffix=".yaml")

    def json_path(self, stage_idx: int = -1, task_idx: int = 0) -> Optional[Path]:
        return self._task_file_path(stage_idx, task_idx, suffix=".json")

    def write_task_json(
        self, data: List[Dict[str, Any]], stage_idx: int = -1, task_idx: int = 0
    ):
        r"""Dump `data` to a new JSON file for the agent.

        Dump `data` to a JSON file written to the directory specified
        by `stage_idx` and `task_idx`.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of dictionarys to pass to `json.dump()`. Values in the
            dictionarys must be JSON serializable.
        """
        path = self.json_path(stage_idx, task_idx)
        assert path is not None
        with open(path, "w") as f:
            json.dump(data, f)

    def read_task_json(
        self, stage_idx: int = -1, task_idx: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        path = self.json_path(stage_idx, task_idx)
        if path is None:
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return data


class DeepDriveMD_API:

    # Directory structure for experiment stages
    MOLECULAR_DYNAMICS_DIR = "molecular_dynamics_runs"
    AGGREGATE_DIR = "aggregation_runs"
    MACHINE_LEARNING_DIR = "machine_learning_runs"
    MODEL_SELECTION_DIR = "model_selection_runs"
    AGENT_DIR = "agent_runs"

    def __init__(self, experiment_directory: PathLike):
        self.experiment_dir = Path(experiment_directory)
        self.molecular_dynamics_stage = self._stage_api(self.MOLECULAR_DYNAMICS_DIR)
        self.aggregation_stage = self._stage_api(self.AGGREGATE_DIR)
        self.machine_learning_stage = self._stage_api(self.MACHINE_LEARNING_DIR)
        self.model_selection_stage = self._stage_api(self.MODEL_SELECTION_DIR)
        self.agent_stage = self._stage_api(self.AGENT_DIR)

    def _stage_api(self, dirname):
        """Factory function for Stage_API."""
        return Stage_API(self.experiment_dir, dirname)

    def get_total_iterations(self):
        return self.molecular_dynamics_stage.stage_dir_count()

    def get_last_n_md_runs(
        self,
        n: Optional[int] = None,
        data_file_suffix: str = ".h5",
        traj_file_suffix: str = ".dcd",
        structure_file_suffix: str = ".pdb",
    ) -> Dict[str, List[str]]:
        r"""Get the last `n` MD run directories data file paths.

        Return a dictionary of data file paths for the last `n` MD runs
        including the training data files, the trajectory files, and the
        coordinate files.

        Parameters
        ----------
        n : int, optional
            Number of latest MD run directories to glob data files from.
            Defaults to all MD run directories.
        data_file_suffix : int, optional
            The suffix of the training data file. Defaults to ".h5".
        traj_file_suffix : str, optional
            The suffix of the traj file. Defaults to ".dcd".
        structure_file_suffix : str, optional
            The suffix of the structure file. Defaults to ".pdb".

        Returns
        -------
        Dict[str, List[str]]
            A dictionary with keys "data_files", "traj_files" and "structure_files"
            each containing a list of `n` paths globed from the the latest `n`
            MD run directories.
        """
        # /self.molecular_dynamics_dir
        #   /stage_{stage_idx}
        #       /task_{task_idx}
        run_dirs = self.molecular_dynamics_stage.runs_dir.glob("*/task*")
        # Remove any potential files
        run_dirs = filter(lambda p: p.is_dir(), run_dirs)
        # Sort by deepdrivemd iteration and sim task id
        run_dirs = sorted(run_dirs)
        # Reverse sort to get last n
        run_dirs = reversed(run_dirs)
        # Evaluate generator up to n items
        run_dirs = list(itertools.islice(run_dirs, n))
        # Put back in sequential order
        run_dirs = reversed(run_dirs)
        # Convert pathlib.Path to str
        run_dirs = list(map(str, run_dirs))

        return {
            "data_files": glob_file_from_dirs(run_dirs, f"*{data_file_suffix}"),
            "traj_files": glob_file_from_dirs(run_dirs, f"*{traj_file_suffix}"),
            "structure_files": glob_file_from_dirs(
                run_dirs, f"*{structure_file_suffix}"
            ),
        }

    def get_restart_pdb(
        self, index: int, stage_idx: int = -1, task_idx: int = 0
    ) -> Dict[str, Any]:
        r"""Gets a single datum for the restart points JSON file.

        Parameters
        ----------
        index : int
            Index into the agent_{}.json file of the latest
            DeepDriveMD iteration.

        Returns
        -------
        Dict[Any]
            Dictionary entry written by the outlier detector.
        """
        data = self.agent_stage.read_task_json(stage_idx, task_idx)
        assert data is not None
        return data[index]

    @staticmethod
    def get_initial_pdbs(initial_pdb_dir: PathLike) -> List[Path]:
        r"""Return a list of PDB paths from the `initial_pdb_dir`.

        Parameters
        ----------
        initial_pdb_dir : Union[str, Path]
            Initial data directory passed containing PDBs and optional topologies.

        Returns
        -------
        List[Path]
            List of paths to initial PDB files.

        Raises
        ------
        ValueError
            If any of the PDB file names contain a double underscore __.
        """

        pdb_filenames = list(Path(initial_pdb_dir).glob("*/*.pdb"))

        if any("__" in filename.as_posix() for filename in pdb_filenames):
            raise ValueError("Initial PDB files cannot contain a double underscore __")

        return pdb_filenames

    @staticmethod
    def get_system_name(pdb_file: PathLike) -> str:
        r"""Parse the system name from a PDB file.

        Parameters
        ----------
        pdb_file : Union[str, Path]
            The PDB file to parse. Can be absolute path,
            relative path, or filename.

        Returns
        -------
        str
            The system name used to identify system topology.

        Examples
        --------
        >>> pdb_file = "/path/to/system_name__anything.pdb"
        >>> DeepDriveMD_API.get_system_name(pdb_file)
        'system_name'

        >>> pdb_file = "/path/to/system_name/anything.pdb"
        >>> DeepDriveMD_API.get_system_name(pdb_file)
        'system_name'
        """
        pdb_file = Path(pdb_file)
        # On subsequent iterations the PDB file names include
        # the system information to look up the topology
        if "__" in pdb_file.name:
            return Path(pdb_file).name.split("__")[0]

        # On initial iterations the system name is the name of the
        # subdirectory containing pdb/top files
        return pdb_file.parent.name

    @staticmethod
    def get_topology(
        initial_pdb_dir: PathLike, pdb_file: PathLike, suffix: str = ".top"
    ) -> Optional[Path]:
        r"""Get the topology file for the system.

        Parse `pdb_file` for the system name and then retrieve
        the topology file from the `initial_pdb_dir` or return None
        if the system doesn't have a topology.

        Parameters
        ----------
        initial_pdb_dir : Union[str, Path]
            Initial data directory passed containing PDBs and optional topologies.
        pdb_file : Union[str, Path]
            The PDB file to parse. Can be absolute path, relative path, or filename.
        suffix : str
            Suffix of the topology file (.top, .prmtop, etc).

        Returns
        -------
        Optional[Path]
            The path to the topology file, or None if system has no topology.

        """
        # pdb_file: /path/to/pdb/<system-name>__<everything-else>.pdb
        # top_file: initial_pdb_dir/<system-name>/*<suffix>
        system_name = DeepDriveMD_API.get_system_name(pdb_file)
        top_file = list(Path(initial_pdb_dir).joinpath(system_name).glob(f"*{suffix}"))
        if top_file:
            return top_file[0]
        return None

    @staticmethod
    def get_system_pdb_name(pdb_file: PathLike) -> str:
        r"""Generate PDB file name with correct system name.

        Parse `pdb_file` for the system name and generate a
        PDB file name that is parseable by DeepDriveMD. If
        `pdb_file` name is already compatible with DeepDriveMD,
        the returned name will be the same.

        Parameters
        ----------
        pdb_file : Union[str, Path]
            The PDB file to parse. Can be absolute path,
            relative path, or filename.

        Returns
        -------
        str
            The new PDB file name. File is not created.

        Raises
        ------
        ValueError
            If `pdb_file` contains more than one __.

        Examples
        --------
        >>> pdb_file = "/path/to/system_name__anything.pdb"
        >>> DeepDriveMD_API.get_system_pdb_name(pdb_file)
        'system_name__anything.pdb'

        >>> pdb_file = "/path/to/system_name/anything.pdb"
        >>> DeepDriveMD_API.get_system_pdb_name(pdb_file)
        'system_name__anything.pdb'
        """
        pdb_file = Path(pdb_file)

        __count = pdb_file.name.count("__")

        if __count == 0:
            return f"{pdb_file.parent.name}__{pdb_file.name}"
        elif __count == 1:
            return pdb_file.name

        raise ValueError(
            f"pdb_file can only have one occurence of __ not {__count}.\n{pdb_file}"
        )

    @staticmethod
    def write_pdb(
        output_pdb_file: PathLike,
        input_pdb_file: PathLike,
        traj_file: PathLike,
        frame: int,
        in_memory: bool = False,
    ):
        r"""Write a PDB file.

        Writes `output_pdb_file` to disk containing coordindates of
        a single `frame` from a given input PDB `input_pdb_file` and
        trajectory file `traj_file`.

        Parameters
        ----------
        output_pdb_file : Union[str, Path]
            The path of the output PDB file to be written to.
        input_pdb_file : Union[str, Path]
            The path of the input PDB file used to open `traj_file`
            in MDAnalysis.Universe().
        traj_file : Union[str, Path]
            The path of the trajectory file to be read from.
        frame : int
            The frame index into `traj_file` used to write `output_pdb_file`.
        in_memory : bool, optional
            If true, will load the MDAnalysis.Universe() trajectory into memory.

        Examples
        --------
        >>> output_pdb_file = "/path/to/output.pdb"
        >>> input_pdb_file = "/path/to/input.pdb"
        >>> traj_file = "/path/to/traj.dcd"
        >>> frame = 10
        >>> DeepDriveMD_API.write_pdb(output_pdb_file, input_pdb_file, traj_file, frame)
        """
        u = MDAnalysis.Universe(
            str(input_pdb_file), str(traj_file), in_memory=in_memory
        )
        u.trajectory[frame]
        PDB = MDAnalysis.Writer(str(output_pdb_file))
        PDB.write(u.atoms)
