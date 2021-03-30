import sys
import time
from inspect import currentframe, getframeinfo
import numpy as np
from typing import Tuple


def setup_mpi_comm(distributed: bool):
    if distributed:
        # get communicator: duplicate from comm world
        from mpi4py import MPI

        return MPI.COMM_WORLD.Dup()
    return None


def setup_mpi(comm=None) -> Tuple[int, int]:
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    return comm_size, comm_rank


def timer(
    label: str, start: int = 1, frameinfo=None
):  # start = 1 - start, start = -1 - stop, start = 0 - neither
    t = time.localtime()
    gps = time.mktime(t)
    readable = time.asctime(t)
    if frameinfo is None:
        frameinfo = getframeinfo(currentframe().f_back)
    fractions = time.perf_counter()
    print(
        f"TLaBeL|{label}|{start}|{gps}|{readable}|{frameinfo.filename}|{frameinfo.lineno}|{fractions}"
    )
    sys.stdout.flush()


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        frameinfo = getframeinfo(currentframe().f_back)
        timer(self.label, 1, frameinfo)
        return self

    def __exit__(self, type, value, traceback):
        frameinfo = getframeinfo(currentframe().f_back)
        timer(self.label, -1, frameinfo)


def bestk(
    a: np.ndarray, k: int, smallest: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the best `k` values and correspdonding indices.

    Parameters
    ----------
    a : np.ndarray
        Array of dim (N,)
    k : int
        Specifies which element to partition upon.
    smallest : bool
        True if the best values are small (or most negative).
        False if the best values are most positive.

    Returns
    -------
    np.ndarray
        Of length `k` containing the `k` smallest values in `a`.
    np.ndarray
        Of length `k` containing indices of input array `a`
        coresponding to the `k` smallest values in `a`.
    """
    # If larger values are considered best, make large values the smallest
    # in order for the sort function to pick the best values.
    arr = a if smallest else -1 * a
    # Only sorts 1 element of `arr`, namely the element that is position
    # k in the sorted array. The elements above and below the kth position
    # are partitioned but not sorted. Returns the indices of the elements
    # on the left hand side of the partition i.e. the top k.
    best_inds = np.argpartition(arr, k)[:k]
    # Get the associated values of the k-partition
    best_values = arr[best_inds]
    # Only sorts an array of size k
    sort_inds = np.argsort(best_values)
    return best_values[sort_inds], best_inds[sort_inds]
