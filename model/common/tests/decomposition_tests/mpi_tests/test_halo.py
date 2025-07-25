# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.common import exceptions
from icon4py.model.common.decomposition import definitions as defs, mpi_decomposition


try:
    import mpi4py  # import mpi4py to check for optional mpi dependency
    import mpi4py.MPI

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

from gt4py.next import common as gtx_common

from icon4py.model.common.decomposition import halo
from icon4py.model.common.grid import (
    base as base_grid,
    grid_manager as gm,
    gridfile as grid_file,
    simple,
    vertical as v_grid,
)
from icon4py.model.testing import datatest_utils as dt_utils, helpers


UGRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R_ugrid.nc"
)
GRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R.nc"
)
backend = None

"""
TESTDATA using the [SimpleGrid](../../../src/icon4py/model/common/grid/simple.py)
The distribution maps all of the 18 cells of the simple grid to ranks 0..3

the dictionaries contain the mapping from rank to global (in the simple grid) index of the dimension:
_CELL_OWN: rank -> owned cells, essentially the inversion of the SIMPLE_DISTRIBUTION
_EDGE_OWN: rank -> owned edges
_VERTEX_OWN: rank -> owned vertices

the decision as to whether a "secondary" dimension (edge, vertices) is owned by a rank are made according to the
rules and conventions described in (../../../src/icon4py/model/common/decomposition/halo.py)


_CELL_FIRST_HALO_LINE:
_CELL_SECON_HALO_LINE:
_EDGE_FIRST_HALO_LINE:
_EDGE_SECOND_HALO_LINE:
_VERTEX_FIRST_HALO_LINE:
_VERTEX_SECOND_HALO_LINE: :mapping of rank to global indices that belongs to a ranks halo lines.
"""

SIMPLE_DISTRIBUTION = np.asarray(
    [
        0,  # 0c
        1,  # 1c
        1,  # 2c
        0,  # 3c
        0,  # 4c
        1,  # 5c
        0,  # 6c
        0,  # 7c
        2,  # 8c
        2,  # 9c
        0,  # 10c
        2,  # 11c
        3,  # 12c
        3,  # 13c
        1,  # 14c
        3,  # 15c
        3,  # 16c
        1,  # 17c
    ]
)
_CELL_OWN = {0: [0, 3, 4, 6, 7, 10], 1: [1, 2, 5, 14, 17], 2: [8, 9, 11], 3: [12, 13, 15, 16]}

_CELL_FIRST_HALO_LINE = {
    0: [1, 11, 13, 9, 2, 15],
    1: [3, 8, 4, 11, 16, 13, 15],
    2: [5, 7, 6, 12, 14],
    3: [9, 10, 17, 14, 0, 1],
}

_CELL_SECOND_HALO_LINE = {
    0: [17, 5, 12, 14, 8, 16],
    1: [0, 7, 6, 9, 10, 12],
    2: [2, 1, 4, 3, 10, 15, 16, 17],
    3: [6, 7, 8, 2, 3, 4, 5, 11],
}

_CELL_HALO = {
    0: _CELL_FIRST_HALO_LINE[0] + _CELL_SECOND_HALO_LINE[0],
    1: _CELL_FIRST_HALO_LINE[1] + _CELL_SECOND_HALO_LINE[1],
    2: _CELL_FIRST_HALO_LINE[2] + _CELL_SECOND_HALO_LINE[2],
    3: _CELL_FIRST_HALO_LINE[3] + _CELL_SECOND_HALO_LINE[3],
}

_EDGE_OWN = {
    0: [1, 5, 12, 13, 14, 9],
    1: [8, 7, 6, 25, 4, 2],
    2: [16, 11, 15, 17, 10, 24],
    3: [19, 23, 22, 26, 0, 3, 20, 18, 21],
}

_EDGE_FIRST_HALO_LINE = {0: [0, 4, 17, 21, 10, 2], 1: [3, 15, 20, 26, 24], 2: [18], 3: []}

_EDGE_SECOND_HALO_LINE = {
    0: [3, 6, 7, 8, 15, 24, 25, 26, 16, 22, 23, 18, 19, 20, 11],
    1: [0, 1, 5, 9, 12, 11, 10, 13, 16, 17, 18, 19, 21, 22, 23],
    2: [2, 9, 12, 4, 8, 7, 14, 21, 13, 19, 20, 22, 23, 25, 26],
    3: [11, 10, 14, 13, 16, 17, 24, 25, 6, 2, 1, 5, 4, 8, 7],
}

_EDGE_THIRD_HALO_LINE = {
    0: [],
    1: [14],
    2: [0, 1, 3, 5, 6],
    3: [9, 12, 15],
}
_EDGE_HALO = {
    0: _EDGE_FIRST_HALO_LINE[0] + _EDGE_SECOND_HALO_LINE[0] + _EDGE_THIRD_HALO_LINE[0],
    1: _EDGE_FIRST_HALO_LINE[1] + _EDGE_SECOND_HALO_LINE[1] + _EDGE_THIRD_HALO_LINE[1],
    2: _EDGE_FIRST_HALO_LINE[2] + _EDGE_SECOND_HALO_LINE[2] + _EDGE_THIRD_HALO_LINE[2],
    3: _EDGE_FIRST_HALO_LINE[3] + _EDGE_SECOND_HALO_LINE[3] + _EDGE_THIRD_HALO_LINE[3],
}

_VERTEX_OWN = {
    0: [4],
    1: [],
    2: [3, 5],
    3: [
        0,
        1,
        2,
        6,
        7,
        8,
    ],
}

_VERTEX_FIRST_HALO_LINE = {
    0: [0, 1, 5, 8, 7, 3],
    1: [1, 2, 0, 5, 3, 8, 6],
    2: [
        6,
        8,
        7,
    ],
    3: [],
}

_VERTEX_SECOND_HALO_LINE = {
    0: [2, 6],
    1: [7, 4],
    2: [4, 0, 2, 1],
    3: [3, 4, 5],
}
_VERTEX_HALO = {
    0: _VERTEX_FIRST_HALO_LINE[0] + _VERTEX_SECOND_HALO_LINE[0],
    1: _VERTEX_FIRST_HALO_LINE[1] + _VERTEX_SECOND_HALO_LINE[1],
    2: _VERTEX_FIRST_HALO_LINE[2] + _VERTEX_SECOND_HALO_LINE[2],
    3: _VERTEX_FIRST_HALO_LINE[3] + _VERTEX_SECOND_HALO_LINE[3],
}

OWNED = {dims.CellDim: _CELL_OWN, dims.EdgeDim: _EDGE_OWN, dims.VertexDim: _VERTEX_OWN}
HALO = {dims.CellDim: _CELL_HALO, dims.EdgeDim: _EDGE_HALO, dims.VertexDim: _VERTEX_HALO}
FIRST_HALO_LINE = {
    dims.CellDim: _CELL_FIRST_HALO_LINE,
    dims.VertexDim: _VERTEX_FIRST_HALO_LINE,
    dims.EdgeDim: _EDGE_FIRST_HALO_LINE,
}
SECOND_HALO_LINE = {
    dims.CellDim: _CELL_SECOND_HALO_LINE,
    dims.VertexDim: _VERTEX_SECOND_HALO_LINE,
    dims.EdgeDim: _EDGE_SECOND_HALO_LINE,
}


@pytest.fixture(scope="session")
def simple_neighbor_tables():
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray for k, v in grid.connectivities.items() if gtx_common.is_neighbor_table(v)
    }
    return neighbor_tables


def grid_file_manager(file: pathlib.Path) -> gm.GridManager:
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), str(file), v_grid.VerticalGridConfig(num_levels=1)
    )
    manager(keep_skip_values=True)
    return manager


@pytest.mark.mpi(min_size=4)
def test_halo_constructor_owned_cells(processor_props, simple_neighbor_tables):  # F811 # fixture
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
        backend=backend,
    )
    my_owned_cells = halo_generator.owned_cells(SIMPLE_DISTRIBUTION)

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(_CELL_OWN[processor_props.rank])
    assert np.setdiff1d(my_owned_cells, _CELL_OWN[processor_props.rank]).size == 0


@pytest.mark.parametrize("processor_props", [True, False], indirect=True)
def test_halo_constructor_validate_number_of_node_mismatch(processor_props, simple_neighbor_tables):
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    distribution = (processor_props.comm_size + 1) * np.ones((num_cells,), dtype=int)
    with pytest.raises(expected_exception=exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            run_properties=processor_props,
            num_levels=1,
        )
        halo_generator(distribution)
    assert "The distribution assumes more nodes than the current run" in e.value.args[0]


@pytest.mark.parametrize("processor_props", [True, False], indirect=True)
def test_halo_constructor_validate_rank_mapping_wrong_shape(
    processor_props, simple_neighbor_tables
):
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    with pytest.raises(exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            run_properties=processor_props,
            num_levels=1,
        )
        halo_generator(np.zeros((num_cells, 3), dtype=int))
    assert f"should have shape ({num_cells},)" in e.value.args[0]


def global_indices(dim: gtx.Dimension) -> np.ndarray:
    mesh = simple.simple_grid()
    return np.arange(mesh.size[dim], dtype=gtx.int32)


@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.mpi(min_size=4)
def test_element_ownership_is_unique(
    dim, processor_props, simple_neighbor_tables
):  # F811 # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
        backend=backend,
    )

    decomposition_info = halo_generator(SIMPLE_DISTRIBUTION)
    owned = decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"\nrank {processor_props.rank} owns {dim} : {owned} ")
    if not mpi4py.MPI.Is_initialized():
        mpi4py.MPI.Init()
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


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
def test_halo_constructor_decomposition_info_global_indices(
    processor_props, simple_neighbor_tables, dim
):  # F811 # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
    )

    decomp_info = halo_generator(SIMPLE_DISTRIBUTION)
    my_halo = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.HALO)
    print(f"rank {processor_props.rank} has halo {dim} : {my_halo}")
    assert my_halo.size == len(HALO[dim][processor_props.rank])
    assert np.setdiff1d(my_halo, HALO[dim][processor_props.rank], assume_unique=True).size == 0
    my_owned = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"rank {processor_props.rank} owns {dim} : {my_owned} ")
    assert_same_entries(dim, my_owned, OWNED, processor_props.rank)


def assert_same_entries(
    dim: gtx.Dimension, my_owned: np.ndarray, reference: dict[gtx.Dimension, dict], rank: int
):
    assert my_owned.size == len(reference[dim][rank])
    assert np.setdiff1d(my_owned, reference[dim][rank], assume_unique=True).size == 0


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_halo_constructor_decomposition_info_halo_levels(
    processor_props, dim, simple_neighbor_tables
):  # F811 # fixture
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
    )
    decomp_info = halo_generator(SIMPLE_DISTRIBUTION)
    my_halo_levels = decomp_info.halo_levels(dim)
    print(f"{dim.value}: rank {processor_props.rank} has halo levels {my_halo_levels} ")
    if dim != dims.EdgeDim:
        assert np.all(
            my_halo_levels != defs.DecompositionFlag.UNDEFINED
        ), (
            "All indices should have a defined DecompositionFlag"
        )  # THIS WILL CURRENTLY FAIL FOR EDGES
    assert np.where(my_halo_levels == defs.DecompositionFlag.OWNED)[0].size == len(
        OWNED[dim][processor_props.rank]
    )
    owned_local_indices = decomp_info.local_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    assert np.all(
        my_halo_levels[owned_local_indices] == defs.DecompositionFlag.OWNED
    ), "owned local indices should have DecompositionFlag.OWNED"
    first_halo_line_local_index = np.where(
        my_halo_levels == defs.DecompositionFlag.FIRST_HALO_LINE
    )[0]
    first_halo_line_global_index = decomp_info.global_index(
        dim, defs.DecompositionInfo.EntryType.ALL
    )[first_halo_line_local_index]
    assert_same_entries(dim, first_halo_line_global_index, FIRST_HALO_LINE, processor_props.rank)
    second_halo_line_local_index = np.where(
        my_halo_levels == defs.DecompositionFlag.SECOND_HALO_LINE
    )[0]
    second_halo_line_global_index = decomp_info.global_index(
        dim, defs.DecompositionInfo.EntryType.ALL
    )[second_halo_line_local_index]
    assert_same_entries(dim, second_halo_line_global_index, SECOND_HALO_LINE, processor_props.rank)


# TODO unused - remove or fix and use?
def icon_distribution(
    props: defs.ProcessProperties, decomposition_info: defs.DecompositionInfo
) -> np.ndarray:
    cell_index = decomposition_info.global_index(
        dims.CellDim, defs.DecompositionInfo.EntryType.OWNED
    )
    comm = props.comm
    local_sizes, recv_buffer = gather_field(cell_index, comm)
    distribution = np.empty((sum(local_sizes)), dtype=int)
    if comm.rank == 0:
        start_index = 0
        for s in comm.size:
            end_index = local_sizes[s]
            distribution[recv_buffer[start_index:end_index]] = s
            start_index = end_index

    comm.Bcast(distribution, root=0)
    return distribution


def gather_field(field: np.ndarray, comm: mpi4py.MPI.Comm) -> tuple:
    local_sizes = np.array(comm.gather(field.size, root=0))
    if comm.rank == 0:
        recv_buffer = np.empty(sum(local_sizes), dtype=field.dtype)
    else:
        recv_buffer = None
    comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    return local_sizes, recv_buffer


def decompose(grid: base_grid.Grid, processor_props):  # F811 # fixture
    partitioner = halo.SimpleMetisDecomposer()
    labels = partitioner(
        grid.connectivities[dims.C2E2C].asnumpy(), n_part=processor_props.comm_size
    )
    return labels


@pytest.mark.xfail
@pytest.mark.mpi
def test_distributed_fields(processor_props):  # F811 # fixture
    grid_manager = grid_file_manager(GRID_FILE)

    global_grid = grid_manager.grid

    global_cell_area = grid_manager.geometry[grid_file.GeometryName.CELL_AREA]
    global_edge_lat = grid_manager.coordinates[dims.EdgeDim]["lat"]
    global_vertex_lon = grid_manager.coordinates[dims.VertexDim]["lon"]

    labels = decompose(global_grid, processor_props)

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=global_grid.neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
    )
    decomposition_info = halo_generator(labels)
    # distributed read: read one field per dimension

    ## TODO why is this local??
    local_cell_area = grid_manager.geometry[grid_file.GeometryName.CELL_AREA]
    local_edge_lat = grid_manager.coordinates[dims.EdgeDim]["lat"]
    local_vertex_lon = grid_manager.coordinates[dims.VertexDim]["lon"]
    print(
        f"rank = {processor_props.rank} has size(cell_area): {local_cell_area.ndarray.shape}, "
        f"has size(edge_length): {local_edge_lat.ndarray.shape}, has size(vertex_length): {local_vertex_lon.ndarray.shape}"
    )
    # the local number of cells must be at most the global number of cells (analytically computed)
    assert (
        local_cell_area.asnumpy().shape[0] <= global_grid.global_properties.num_cells
    ), "local field is larger than global field"
    # global read: read the same (global fields)

    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.CellDim, global_cell_area, local_cell_area
    )

    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.EdgeDim, global_edge_lat, local_edge_lat
    )
    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.VertexDim, global_vertex_lon, local_vertex_lon
    )


def assert_gathered_field_against_global(
    decomposition_info: defs.DecompositionInfo,
    processor_props: defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
):
    assert (
        local_field.size
        == decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.ALL).size
    )
    owned_entries = local_field[
        decomposition_info.local_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props.comm)
    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED),
        processor_props.comm,
    )
    if processor_props.rank == 0:
        assert np.all(gathered_sizes == global_index_sizes)
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)
        sorted_[gathered_global_indices] = gathered_field
        assert helpers.dallclose(sorted_, global_reference_field)


# TODO add test including halo access:
#  Will uses geofac_div and geofac_n2s


@pytest.mark.xfail
def test_halo_neighbor_access_c2e():
    pytest.fail("TODO implement")
    # geofac_div = primal_edge_length(C2E) * edge_orientation / area

    # 1. read grid and distribue - GridManager

    # 2. get geometry fields (from GridManger) primal_edge_length, edge_orientation, area (local read)
    # 3. compute geofac_div = primal_edge_length * edge_orientation / area
    # 4. gather geofac_div
    # 5 compare (possible reorder


def test_no_halo():
    grid_size = base_grid.HorizontalGridSize(num_cells=9, num_edges=14, num_vertices=6)
    halo_generator = halo.NoHalos(horizontal_size=grid_size, num_levels=10, backend=None)
    decomposition = halo.SingleNodeDecomposer()
    decomposition_info = halo_generator(decomposition(np.arange(grid_size.num_cells), 1))
    # cells
    np.testing.assert_allclose(
        np.arange(grid_size.num_cells), decomposition_info.global_index(dims.CellDim)
    )
    assert np.all(decomposition_info.owner_mask(dims.CellDim))
    assert np.all(decomposition_info.halo_levels(dims.CellDim) == defs.DecompositionFlag.OWNED)
    # edges
    np.testing.assert_allclose(
        np.arange(grid_size.num_edges), decomposition_info.global_index(dims.EdgeDim)
    )
    assert np.all(decomposition_info.halo_levels(dims.EdgeDim) == defs.DecompositionFlag.OWNED)
    assert np.all(decomposition_info.owner_mask(dims.EdgeDim))
    # vertices
    np.testing.assert_allclose(
        np.arange(grid_size.num_vertices), decomposition_info.global_index(dims.VertexDim)
    )
    assert np.all(decomposition_info.halo_levels(dims.VertexDim) == defs.DecompositionFlag.OWNED)
    assert np.all(decomposition_info.owner_mask(dims.VertexDim))
