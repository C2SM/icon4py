from enum import Enum
from typing import Union

import mpi4py
import numpy as np
import numpy.ma as ma
from gt4py.next.common import Dimension
from mpi4py.MPI import Comm


from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.diffusion.utils import builder
mpi4py.rc.initialize = False

CommId = Union[int, Comm, None]

def get_processor_properties(comm_id: CommId = None):
    init_mpi()

    def _get_current_comm_or_comm_world(comm_id: CommId)->Comm:
        if isinstance(comm_id, int):
            comm = Comm.f2py(comm_id)
        elif isinstance(comm_id, Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    current_comm = _get_current_comm_or_comm_world(comm_id)
    return ProcessProperties.from_mpi_comm(current_comm)


def init_mpi():
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()




class DecompositionInfo:
    class EntryType(int, Enum):
        ALL = 0,
        OWNED = 1,
        HALO = 2
    @builder
    def with_dimension(self, dim:Dimension, global_index:np.ndarray, owner_mask:np.ndarray):
        masked_global_index = ma.array(global_index, mask=owner_mask)
        self._global_index[dim] = masked_global_index

    def __init__(self):
        self._global_index = {}



    def global_index(self, dim:Dimension, type:EntryType = EntryType.ALL):
        match(type):
            case DecompositionInfo.EntryType.ALL:
                return ma.getdata(self._global_index[dim], subok=False)
            case _:
                raise NotImplementedError()


