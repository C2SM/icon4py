import pathlib

import mpi4py
import pytest

from icon4py.common.dimension import CellDim
from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.driver.io_utils import SerializationType, read_decomp_info, read_grid
from icon4py.driver.parallel_setup import get_processor_properties, DecompositionInfo

"""
running tests with mpi:

mpirun -np 2 python -m pytest --with-mpi tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""




@pytest.mark.mpi
def test_processor_properties_from_comm_world(mpi):
    props = get_processor_properties()
    assert props.rank < mpi.COMM_WORLD.Get_size()
    assert props.comm_name == mpi.COMM_WORLD.Get_name()


def test_decomposition_info():
    path = pathlib.Path(
        "/home/magdalena/data/exclaim/dycore/mch_ch_r04b09_dsl/node2/mch_ch_r04b09_dsl/icon_grid")
    props = get_processor_properties()
    decomposition_info = read_decomp_info(path, props, SerializationType.SB)
    icon_grid = read_grid(path, props, SerializationType.SB)
    global_index = decomposition_info.global_index(CellDim, DecompositionInfo.EntryType.ALL)
    assert global_index.shape[0] == icon_grid.num_cells()
