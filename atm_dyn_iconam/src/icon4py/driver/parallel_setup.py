from typing import Optional, Union

import mpi4py
from mpi4py.MPI import Comm


from icon4py.decomposition.decomposed import ProcessProperties

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
