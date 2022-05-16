from pathlib import Path
from typing import Dict

import adios2
import numpy as np
import torch
from torch.utils.data import Dataset


class CenterOfMassTransform:
    def __init__(self, data: np.ndarray) -> None:
        """Computes center of mass transformation
        Parameters
        ----------
        data : np.ndarray
            Dataset of positions with shape (num_examples, 3, num_points).
        """

        # Center of mass over points
        cms = np.mean(data.astype(np.float64), axis=2, keepdims=True).astype(np.float32)
        # Scalar bias and scale normalization factors
        self.bias: float = (data - cms).min()
        self.scale: float = 1.0 / ((data - cms).max() - self.bias)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize example by bias and scale factors
        Parameters
        ----------
        x : np.ndarray
            Data to transform shape (3, num_points). Modifies :obj:`x`.
        Returns
        -------
        np.ndarray
            The transformed data
        Raises
        ------
        ValueError
            If NaN encountered in input
        """
        x -= np.mean(x, axis=1, keepdims=True)

        if np.any(np.isnan(x)):
            raise ValueError("NaN encountered in input.")

        # Normalize
        x = (x - self.bias) * self.scale
        return x


class PointCloudDatasetInMemory(Dataset):
    """
    PyTorch Dataset class to load point cloud data. Optionally, uses HDF5
    files to only read into memory what is necessary for one batch.
    """

    def __init__(
        self,
        data: np.ndarray,
        scalars: Dict[str, np.ndarray] = {},
        cms_transform: bool = False,
        scalar_requires_grad: bool = False,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Dataset of positions with shape (num_examples, 3, num_points)
        scalars : Dict[str, np.ndarray], default={}
            Dictionary of scalar arrays. For instance, the root mean squared
            deviation (RMSD) for each feature vector can be passed via
            :obj:`{"rmsd": np.array(...)}`. The dimension of each scalar array
            should match the number of input feature vectors N.
        cms_transform: bool
            If True, subtract center of mass from batch and shift and scale
            batch by the full dataset statistics.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            :obj:`scalar_dset_names`. Set to True, to use scalars for learning.
            If scalars are only required for plotting, then set it as False.
        """
        self.data = data
        self.scalars = scalars
        self.cms_transform = cms_transform
        self.scalar_requires_grad = scalar_requires_grad

        if self.cms_transform:
            self.transform = CenterOfMassTransform(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        data = self.data[idx].copy()  # shape (3, num_points)

        # CMS subtract
        if self.cms_transform:
            data = self.transform.transform(data)

        sample = {"X": torch.from_numpy(data)}
        # Add scalars
        for name, dset in self.scalars.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self.scalar_requires_grad
            )
        return sample


def read_adios_file(input_path: Path):
    with adios2.open(str(input_path), "r") as fr:
        n = fr.steps()

        shape = list(
            map(
                int,
                fr.available_variables()["point_cloud"]["Shape"]
                .replace(",", "")
                .split(),
            )
        )

        points = fr.read("point_cloud", [0, 0], shape, 0, n)

    return points
