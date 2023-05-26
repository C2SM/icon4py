import mpi4py.MPI


class ProcessProperties():
    def __init__(self, name='', rank = 0):
        self._communicator_name: str = name
        self._rank: int = rank

    def rank(self):
        return self._rank

    def comm_name(self):
        return self._communicator_name
    @classmethod
    def from_mpi_comm(cls, comm: mpi4py.MPI.Comm):
        return ProcessProperties(comm.Get_name(), comm.Get_rank())


