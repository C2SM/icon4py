# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.common import exceptions
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.testing import parallel_helpers
from icon4py.model.testing.fixtures import processor_props

from ..fixtures import simple_neighbor_tables


try:
    import mpi4py  # import mpi4py to check for optional mpi dependency
    import mpi4py.MPI

    from icon4py.model.common.decomposition import mpi_decomposition

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

from icon4py.model.common.decomposition import halo
from icon4py.model.common.grid import base as base_grid, simple

from .. import utils


backend = None


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_halo_constructor_owned_cells(processor_props, simple_neighbor_tables):  # F811 # fixture
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        allocator=backend,
    )
    my_owned_cells = halo_generator.owned_cells(utils.SIMPLE_DISTRIBUTION)

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(utils._CELL_OWN[processor_props.rank])
    assert np.setdiff1d(my_owned_cells, utils._CELL_OWN[processor_props.rank]).size == 0


@pytest.mark.parametrize("processor_props", [True, False], indirect=True)
def test_halo_constructor_validate_number_of_node_mismatch(processor_props, simple_neighbor_tables):
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    distribution = (processor_props.comm_size + 1) * np.ones((num_cells,), dtype=int)
    with pytest.raises(expected_exception=exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            run_properties=processor_props,
        )
        halo_generator(distribution)
    assert "The distribution assumes more nodes than the current run" in e.value.args[0]


def global_indices(dim: gtx.Dimension) -> np.ndarray:
    mesh = simple.simple_grid()
    return np.arange(mesh.size[dim], dtype=gtx.int32)


@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_element_ownership_is_unique(
    dim,
    processor_props,
    simple_neighbor_tables,
):
    parallel_helpers.check_comm_size(processor_props, sizes=[4])

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        allocator=backend,
    )

    decomposition_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    owned = decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"\nrank {processor_props.rank} owns {dim} : {owned} ")
    # assert that each cell is only owned by one rank
    comm = processor_props.comm

    my_size = owned.shape[0]
    local_sizes = np.array(comm.gather(my_size, root=0))
    buffer_size = 27
    send_buf = -1 * np.ones(buffer_size, dtype=int)
    send_buf[:my_size] = owned
    print(f"rank {processor_props.rank} send_buf: {send_buf}")
    if processor_props.rank == 0:
        print(f"local_sizes: {local_sizes}")
        recv_buffer = -1 * np.ones((4, buffer_size), dtype=int)
        print(f"{recv_buffer.shape}")
    else:
        recv_buffer = None
    # Gatherv does not work if one of the buffers has size-0 (VertexDim)
    comm.Gather(sendbuf=send_buf, recvbuf=recv_buffer, root=0)
    if processor_props.rank == 0:
        print(f"global indices: {recv_buffer}")
        # check there are no duplicates
        values = recv_buffer[recv_buffer != -1]
        assert values.size == len(np.unique(values))
        # check the buffer has all global indices
        assert np.all(np.sort(values) == global_indices(dim))


def decompose(grid: base_grid.Grid, processor_props):
    partitioner = halo.SimpleMetisDecomposer()
    labels = partitioner(
        grid.connectivities[dims.C2E2C].asnumpy(), n_part=processor_props.comm_size
    )
    return labels
