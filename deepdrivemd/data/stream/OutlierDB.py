import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union


class OutlierDB:
    """Stores the metadata for outliers to be used by simulations.

    Attributes
    ----------
    dir : Path
          directory with published outliers
    sorted_index: List[str]
          list of md5sums of outlier positions (used as a name of
          an outlier pdb or numpy file) sorted by the corresponding rmsd
    dictionary: Dict
          maps md5sum to rmsd
    """

    def __init__(self, dir: Path, restarts: List[Tuple[List[float], str]]):
        """Constructor

        Parameters
        ----------
        dir : Path
              directory with published outliers
        restarts : List[Tuple[float, str]]
              list of outliers given as tuples of rmsd and md5sum of positions (used as a file name)
        """
        self.dir = dir
        self.sorted_index = list(
            map(lambda x: os.path.basename(x[1]).replace(".pdb", ""), restarts)
        )
        self.dictionary: Dict[str, List[float]] = {}
        for rmsd, path in restarts:
            md5 = os.path.basename(path).replace(".pdb", "")
            self.dictionary[md5] = rmsd
        self.print()

    def print(self, n: int = 5) -> None:
        print("=" * 30)
        print("In OutlierDB")
        n = min(n, len(self.sorted_index))
        for i in range(n):
            md5 = self.sorted_index[i]
            print(f"{md5}: {self.dictionary[md5]}")
        print("=" * 30)

    def next_random(
        self, m: Union[int, None] = None, alpha: int = 1, beta: int = 25
    ) -> str:
        """Return next outlier using beta distribution that prefers smaller rmsds

        Parameters
        ----------
        m : int, default = -1
            if `m` is not `None`, restrict the random selection to the first
            `m` elements of `softed_index`, otherwise - any element can be chosen.
        alpha : int, default = 1
        beta : int, default = 25
            `alpha` and `beta` are parameters of beta distribution.
        """
        if len(self.sorted_index) == 0:
            raise ValueError("len(sorted_index) = 0")
        if m is None:
            hlimit = len(self.sorted_index) - 1
        else:
            hlimit = min(m, len(self.sorted_index) - 1)
        i = int(random.betavariate(alpha=alpha, beta=beta) * (hlimit))
        md5 = self.sorted_index[i]

        selected_rmsd = self.dictionary[md5][0]
        rmsds = list(map(lambda x: x[0], self.dictionary.values()))
        min_rmsd = min(rmsds)
        max_rmsd = max(rmsds)
        print(
            f"In next_random: selected_rmsd = {selected_rmsd}, min_rmsd = {min_rmsd}, max_rmsd = {max_rmsd}, md5 = {md5}, index = {i}, len = {len(self.sorted_index)}"
        )

        return f"{self.dir}/{md5}.pdb"
