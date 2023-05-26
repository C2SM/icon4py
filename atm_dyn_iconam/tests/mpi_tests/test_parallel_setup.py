import mpi4py
import pytest

from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.driver.parallel_setup import get_processor_properties

"""
running tests with mpi:

mpirun -np 2 python -m pytest --with-mpi tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""

@pytest.fixture
def mpi():
    from mpi4py import MPI
    mpi4py.rc.initialize = False
    return MPI

@pytest.mark.mpi()
def test_processor_properties_from_comm_world(mpi):
    props = get_processor_properties()
    assert props.rank() < mpi.COMM_WORLD.Get_size()
    assert props.comm_name() == mpi.COMM_WORLD.Get_name()
