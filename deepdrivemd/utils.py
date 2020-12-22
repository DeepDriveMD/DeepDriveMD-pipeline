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


def topk(a, k):
    r"""Return the top k indices of a kth element sort.

    Only sorts 1 element of `a`, namely the element that is position
    k in the sorted array. The elements above and below the kth position
    are partitioned but not sorted. Returns the indices of the elements
    on the left hand side of the partition i.e. the top k.

    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)
    k : intå
        specifies which element to partition upon

    Returns
    -------
    np.ndarray
        Of length k containing indices of input array a
        coresponding to the k smallest values in a.
    """
    return np.argpartition(a, k)[:k]
