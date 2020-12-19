import json
import itertools
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import MDAnalysis

PathLike = Union[str, Path]


def glob_file_from_dirs(dirs: List[str], pattern: str) -> List[str]:
    """Return a list of all items matching `pattern` in multiple `dirs`."""
    return [next(Path(d).glob(pattern)).as_posix() for d in dirs]


class DeepDriveMD_API:

    # Directory structure for experiment
    MOLECULAR_DYNAMICS_DIR = "molecular_dynamics_runs"
    AGGREGATE_DIR = "aggregation_runs"
    MACHINE_LEARNING_DIR = "machine_learning_runs"
    MODEL_SELECTION_DIR = "model_selection_runs"
    AGENT_DIR = "agent_runs"
    TMP_DIR = "tmp"

    # File name and subdirectory prefixes
    AGGREGATION_PREFIX = "aggregation_"
    ML_PREFIX = "ml_"
    MODEL_SELECTION_PREFIX = "modelselection_"
    AGENT_PREFIX = "agent_"

    @staticmethod
    def idx_label(idx):
        return f"{idx:04d}"

    @staticmethod
    def get_idx_label(path: Path) -> str:
        return path.with_suffix("").name.split("_")[-1]

    @staticmethod
    def get_latest(path: Path, pattern: str, is_dir=False) -> Optional[Path]:
        # Assumes file has format XXX_YYY_..._{iteration:03d}.ZZZ
        matches = list(filter(lambda p: p.is_dir() == is_dir, path.glob(pattern)))
        if not matches:
            return None
        return max(matches, key=DeepDriveMD_API.get_idx_label)

    @staticmethod
    def next_idx(path: Path, pattern: str) -> int:
        latest = DeepDriveMD_API.get_latest(path, pattern)
        assert latest is not None
        idx = int(DeepDriveMD_API.get_idx_label(latest))
        return idx

    def __init__(self, experiment_directory: PathLike):
        self.experiment_dir = Path(experiment_directory)

    @property
    def molecular_dynamics_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.MOLECULAR_DYNAMICS_DIR)

    @property
    def aggregation_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.AGGREGATE_DIR)

    @property
    def machine_learning_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.MACHINE_LEARNING_DIR)

    @property
    def model_selection_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.MODEL_SELECTION_DIR)

    @property
    def agent_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.AGENT_DIR)

    @property
    def tmp_dir(self) -> Path:
        return self.experiment_dir.joinpath(self.TMP_DIR)

    def aggregation_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the aggregated HDF5 path for a given `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to HDF5 file containing aggregated data. or
            `None` if `iteration == -1` and no HDF5 files exist.
        """
        if iteration == -1:
            return self.get_latest(
                self.aggregation_dir, f"{self.AGGREGATION_PREFIX}*.h5"
            )
        return self.aggregation_dir.joinpath(
            f"{self.AGGREGATION_PREFIX}{self.idx_label(iteration)}.h5"
        )

    def machine_learning_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the ML model path for a given `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to model directory containing ML run. or `None` if
            `iteration == -1` and no machine learning directories exist.
        """
        if iteration == -1:
            return self.get_latest(
                self.machine_learning_dir, f"{self.ML_PREFIX}*", is_dir=True
            )
        return self.machine_learning_dir.joinpath(
            f"{self.ML_PREFIX}{self.idx_label(iteration)}"
        )

    def agent_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the agent path for a given `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to agent directory containing agent run. or
            `None` if `iteration == -1` and no agent directories exist.
        """
        if iteration == -1:
            return self.get_latest(self.agent_dir, f"{self.AGENT_PREFIX}*", is_dir=True)
        return self.agent_dir.joinpath(
            f"{self.AGENT_PREFIX}{self.idx_label(iteration)}"
        )

    def aggregation_config_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the aggregation config file path for a given `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to yaml file containing aggregation config or
            `None` if `iteration == -1` and no yaml files exist.
        """
        if iteration == -1:
            return self.get_latest(
                self.aggregation_dir, f"{self.AGGREGATION_PREFIX}*.yaml"
            )
        return self.aggregation_dir.joinpath(
            f"{self.AGGREGATION_PREFIX}{self.idx_label(iteration)}.yaml"
        )

    def machine_learning_config_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the machine learning config file path for a given `iteration`,

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to yaml file containing machine learning config or
            `None` if `iteration == -1` and no yaml files exist.
        """
        if iteration == -1:
            return self.get_latest(self.machine_learning_dir, f"{self.ML_PREFIX}*.yaml")
        return self.machine_learning_dir.joinpath(
            f"{self.ML_PREFIX}{self.idx_label(iteration)}.yaml"
        )

    def model_selection_config_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the model selection config file path for a given `iteration`,

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to yaml file containing machine learning config or
            `None` if `iteration == -1` and no yaml files exist.
        """
        if iteration == -1:
            return self.get_latest(
                self.model_selection_dir, f"{self.MODEL_SELECTION_PREFIX}*.yaml"
            )
        return self.model_selection_dir.joinpath(
            f"{self.MODEL_SELECTION_PREFIX}{self.idx_label(iteration)}.yaml"
        )

    def agent_config_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Return the agent config file path for a given `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to yaml file containing agent config or
            `None` if `iteration == -1` and no yaml files exist.
        """
        if iteration == -1:
            return self.get_latest(self.agent_dir, f"{self.AGENT_PREFIX}*.yaml")
        return self.agent_dir.joinpath(
            f"{self.AGENT_PREFIX}{self.idx_label(iteration)}.yaml"
        )

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
        # Run dirs: f"run{deepdrivemd_iteration:03d}_{sim_task_id:04d}"
        run_dirs = self.molecular_dynamics_dir.glob("*")
        # Remove any potential files
        run_dirs = filter(lambda x: x.is_dir(), run_dirs)
        # Convert pathlib.Path to str
        run_dirs = map(lambda x: x.as_posix(), run_dirs)
        # Sort by deepdrivemd iteration and sim task id
        run_dirs = sorted(run_dirs)
        # Reverse sort to get last n
        run_dirs = reversed(run_dirs)
        # Evaluate generator up to n items
        run_dirs = list(itertools.islice(run_dirs, n))

        return {
            "data_files": glob_file_from_dirs(run_dirs, f"*{data_file_suffix}"),
            "traj_files": glob_file_from_dirs(run_dirs, f"*{traj_file_suffix}"),
            "structure_files": glob_file_from_dirs(
                run_dirs, f"*{structure_file_suffix}"
            ),
        }

    def get_model_selection_json_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Get the JSON path written by the model selection at `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to JSON file containing model selection metadata or
            `None` if `iteration == -1` and no JSON files exist.
        """
        if iteration == -1:
            return self.get_latest(
                self.model_selection_dir, f"{self.MODEL_SELECTION_PREFIX}*.json"
            )
        return self.model_selection_dir.joinpath(
            f"{self.MODEL_SELECTION_PREFIX}{self.idx_label(iteration)}.json"
        )

    def write_model_selection_json(self, data: List[Dict[str, Any]]):
        r"""Dump `data` to a new JSON file for the agent.

        Dump `data` to a JSON file with the file name in increasing order
        from previous calls to `write_model_selection_json`.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of dictionarys to pass to `json.dump()`. Values in the
            dictionarys must be JSON serializable.
        """
        idx = self.next_idx(
            self.model_selection_dir, f"{self.MODEL_SELECTION_PREFIX}*.json"
        )
        new_restart_path = self.get_model_selection_json_path(idx)
        assert new_restart_path is not None
        with open(new_restart_path, "w") as f:
            json.dump(data, f)

    def get_agent_json_path(self, iteration: int = -1) -> Optional[Path]:
        r"""Get the JSON path written by the agent at `iteration`.

        Parameters
        ----------
        iteration : int
            Iteration of DeepDriveMD. Defaults to most recently created.

        Returns
        -------
        Path, optional
            Path to JSON file containing agent metadata or
            `None` if `iteration == -1` and no JSON files exist.
        """
        if iteration == -1:
            return self.get_latest(self.agent_dir, f"{self.AGENT_PREFIX}*.json")
        return self.agent_dir.joinpath(
            f"{self.AGENT_PREFIX}{self.idx_label(iteration)}.json"
        )

    def write_agent_json(self, data: List[Dict[str, Any]]):
        r"""Dump `data` to a new JSON file for the agent.

        Dump `data` to a JSON file with the file name in increasing order
        from previous calls to `write_agent_json`.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of dictionarys to pass to `json.dump()`. Values in the
            dictionarys must be JSON serializable.
        """
        idx = self.next_idx(self.agent_dir, f"{self.AGENT_PREFIX}*.json")
        new_restart_path = self.get_agent_json_path(idx)
        assert new_restart_path is not None
        with open(new_restart_path, "w") as f:
            json.dump(data, f)

    def get_restart_pdb(self, index: int) -> Dict[str, Any]:
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
        path = self.get_agent_json_path()
        assert path is not None
        with open(path, "r") as f:
            return json.load(f)[index]

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
    def get_topology(initial_pdb_dir: PathLike, pdb_file: PathLike) -> Optional[Path]:
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

        Returns
        -------
        Optional[Path]
            The path to the topology file, or None if system has no topology.

        """
        # pdb_file: /path/to/pdb/<system-name>__<everything-else>.pdb
        # top_file: initial_pdb_dir/<system-name>/*.top
        system_name = DeepDriveMD_API.get_system_name(pdb_file)
        top_file = list(Path(initial_pdb_dir).joinpath(system_name).glob("*.top"))
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
