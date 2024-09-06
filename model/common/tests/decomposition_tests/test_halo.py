# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pathlib

import gt4py.next as gtx
import mpi4py
import mpi4py.MPI
import numpy as np
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.grid_manager as gm
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.decomposition.definitions import DecompositionFlag
from icon4py.model.common.decomposition.halo import (
    HaloGenerator,
    SimpleMetisDecomposer,
)
from icon4py.model.common.grid import base as base_grid, simple, vertical as v_grid
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)


UGRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R_ugrid.nc"
)
GRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R.nc"
)

SIMPLE_DISTRIBUTION = xp.asarray(
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


def test_halo_constructor_owned_cells(processor_props):  # noqa F811 # fixture
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        connectivities=grid.connectivities,
        run_properties=processor_props,
        rank_mapping=SIMPLE_DISTRIBUTION,
        num_levels=1,
    )
    my_owned_cells = halo_generator.owned_cells()

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(_CELL_OWN[processor_props.rank])
    assert xp.setdiff1d(my_owned_cells, _CELL_OWN[processor_props.rank]).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.mpi(min_size=4)
def test_element_ownership_is_unique(dim, processor_props):  # noqa F811 # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        connectivities=grid.connectivities,
        run_properties=processor_props,
        rank_mapping=SIMPLE_DISTRIBUTION,
        num_levels=1,
    )

    decomposition_info = halo_generator()
    owned = decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"\nrank {processor_props.rank} owns {dim} : {owned} ")
    if not mpi4py.MPI.Is_initialized():
        mpi4py.MPI.Init()
    # assert that each cell is only owned by one rank
    comm = processor_props.comm

    my_size = owned.shape[0]
    local_sizes = xp.array(comm.gather(my_size, root=0))
    buffer_size = 27
    send_buf = -1 * xp.ones(buffer_size, dtype=int)
    send_buf[:my_size] = owned
    print(f"rank {processor_props.rank} send_buf: {send_buf}")
    if processor_props.rank == 0:
        print(f"local_sizes: {local_sizes}")
        recv_buffer = -1 * xp.ones((4, buffer_size), dtype=int)
        print(f"{recv_buffer.shape}")
    else:
        recv_buffer = None
    # Gatherv does not work if one of the buffers has size-0 (VertexDim)
    comm.Gather(sendbuf=send_buf, recvbuf=recv_buffer, root=0)
    if processor_props.rank == 0:
        print(f"global indices: {recv_buffer}")
        # check there are no duplicates
        values = recv_buffer[recv_buffer != -1]
        assert values.size == len(xp.unique(values))
        # check the buffer has all global indices
        assert xp.all(xp.sort(values) == global_indices(dim))


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
def test_halo_constructor_decomposition_info_global_indices(processor_props, dim):  # noqa F811 # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        connectivities=grid.connectivities,
        run_properties=processor_props,
        rank_mapping=SIMPLE_DISTRIBUTION,
        num_levels=1,
    )

    decomp_info = halo_generator()
    my_halo = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.HALO)
    print(f"rank {processor_props.rank} has halo {dim} : {my_halo}")
    assert my_halo.size == len(HALO[dim][processor_props.rank])
    assert xp.setdiff1d(my_halo, HALO[dim][processor_props.rank], assume_unique=True).size == 0
    my_owned = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"rank {processor_props.rank} owns {dim} : {my_owned} ")
    assert_same_entries(dim, my_owned, OWNED, processor_props.rank)


def assert_same_entries(
    dim: gtx.Dimension, my_owned: np.ndarray, reference: dict[int, list], rank: int
):
    assert my_owned.size == len(reference[dim][rank])
    assert xp.setdiff1d(my_owned, reference[dim][rank], assume_unique=True).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
def test_halo_constructor_decomposition_info_halo_levels(processor_props, dim):  # noqa F811 # fixture
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        connectivities=grid.connectivities,
        run_properties=processor_props,
        rank_mapping=SIMPLE_DISTRIBUTION,
        num_levels=1,
    )
    decomp_info = halo_generator()
    my_halo_levels = decomp_info.halo_levels(dim)
    print(f"{dim.value}: rank {processor_props.rank} has halo levels {my_halo_levels} ")
    if dim != dims.EdgeDim:
        assert xp.all(
            my_halo_levels != DecompositionFlag.UNDEFINED
        ), (
            "All indices should have a defined DecompositionFlag"
        )  # THIS WILL CURRENTLY FAIL FOR EDGES
    assert xp.where(my_halo_levels == DecompositionFlag.OWNED)[0].size == len(
        OWNED[dim][processor_props.rank]
    )
    owned_local_indices = decomp_info.local_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    assert xp.all(
        my_halo_levels[owned_local_indices] == DecompositionFlag.OWNED
    ), "owned local indices should have DecompositionFlag.OWNED"
    first_halo_line_local_index = xp.where(my_halo_levels == DecompositionFlag.FIRST_HALO_LINE)[0]
    first_halo_line_global_index = decomp_info.global_index(
        dim, defs.DecompositionInfo.EntryType.ALL
    )[first_halo_line_local_index]
    assert_same_entries(dim, first_halo_line_global_index, FIRST_HALO_LINE, processor_props.rank)
    second_halo_line_local_index = xp.where(my_halo_levels == DecompositionFlag.SECOND_HALO_LINE)[0]
    second_halo_line_global_index = decomp_info.global_index(
        dim, defs.DecompositionInfo.EntryType.ALL
    )[second_halo_line_local_index]
    assert_same_entries(dim, second_halo_line_global_index, SECOND_HALO_LINE, processor_props.rank)


def grid_file_manager(file: pathlib.Path) -> gm.GridManager:
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), str(file), v_grid.VerticalGridConfig(num_levels=1)
    )
    manager()
    return manager


def global_indices(dim: dims.Dimension) -> int:
    mesh = simple.SimpleGrid()
    return xp.arange(mesh.size[dim], dtype=xp.int32)


# TODO unused - remove or fix and use?
def icon_distribution(
    props: defs.ProcessProperties, decomposition_info: defs.DecompositionInfo
) -> xp.ndarray:
    cell_index = decomposition_info.global_index(
        dims.CellDim, defs.DecompositionInfo.EntryType.OWNED
    )
    comm = props.comm
    local_sizes, recv_buffer = gather_field(cell_index, comm)
    distribution = xp.empty((sum(local_sizes)), dtype=int)
    if comm.rank == 0:
        start_index = 0
        for s in comm.size:
            end_index = local_sizes[s]
            distribution[recv_buffer[start_index:end_index]] = s
            start_index = end_index

    comm.Bcast(distribution, root=0)
    return distribution


def gather_field(field: xp.ndarray, comm: mpi4py.MPI.Comm) -> tuple:
    local_sizes = xp.array(comm.gather(field.size, root=0))
    if comm.rank == 0:
        recv_buffer = xp.empty(sum(local_sizes), dtype=field.dtype)
    else:
        recv_buffer = None
    comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    return local_sizes, recv_buffer


@pytest.mark.mpi
def test_distributed_fields(processor_props):  # noqa F811 # fixture
    grid_manager = grid_file_manager(GRID_FILE)

    global_grid = grid_manager.grid
    labels = decompose(global_grid, processor_props)

    halo_generator = HaloGenerator(
        connectivities=global_grid.connectivities,
        run_properties=processor_props,
        rank_mapping=labels,
        num_levels=1,
    )
    decomposition_info = halo_generator()
    # distributed read: read one field per dimension
    local_geometry_fields = grid_manager._read_geometry(decomposition_info)
    local_cell_area = local_geometry_fields[gm.GridFile.GeometryName.CELL_AREA]
    local_edge_length = local_geometry_fields[gm.GridFile.GeometryName.EDGE_LENGTH]
    local_vlon = grid_manager.read_coordinates(decomposition_info)[
        gm.GridFile.CoordinateName.VERTEX_LONGITUDE
    ]
    print(
        f"rank = {processor_props.rank} has size(cell_area): {local_cell_area.shape}, has size(edge_length): {local_edge_length.shape}"
    )
    # the local number of cells must be at most the global number of cells (analytically computed)
    assert local_cell_area.size <= global_grid.global_properties.num_cells
    # global read: read the same (global fields)
    global_geometry_fields = grid_manager._read_geometry()
    global_cell_area = global_geometry_fields[gm.GridFile.GeometryName.CELL_AREA]
    global_edge_length = global_geometry_fields[gm.GridFile.GeometryName.EDGE_LENGTH]
    global_vlon = grid_manager.read_coordinates()[gm.GridFile.CoordinateName.VERTEX_LONGITUDE]

    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.CellDim, global_cell_area, local_cell_area
    )

    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.EdgeDim, global_edge_length, local_edge_length
    )
    assert_gathered_field_against_global(
        decomposition_info, processor_props, dims.VertexDim, global_vlon, local_vlon
    )


def decompose(grid: base_grid.BaseGrid, processor_props):  # noqa F811 # fixture
    partitioner = SimpleMetisDecomposer()
    labels = partitioner(grid.connectivities[dims.C2E2CDim], n_part=processor_props.comm_size)
    return labels


def assert_gathered_field_against_global(
    decomposition_info: defs.DecompositionInfo,
    processor_props: defs.ProcessProperties,  # noqa F811 # fixture
    dim: dims.Dimension,
    global_reference_field: xp.ndarray,
    local_field: xp.ndarray,
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
        assert xp.all(gathered_sizes == global_index_sizes)
        sorted_ = xp.zeros(global_reference_field.shape, dtype=xp.float64)
        sorted_[gathered_global_indices] = gathered_field
        assert helpers.dallclose(sorted_, global_reference_field)


# TODO add test including halo access:
#  Will uses geofac_div and geofac_n2s


def test_halo_neighbor_access_c2e():
    ...
    # geofac_div = primal_edge_length(C2E) * edge_orientation / area

    # 1. read grid and distribue - GridManager

    # 2. get geometry fields (from GridManger) primal_edge_length, edge_orientation, area (local read)
    # 3. compute geofac_div = primal_edge_length * edge_orientation / area
    # 4. gather geofac_div
    # 5 compare (possible reorder
