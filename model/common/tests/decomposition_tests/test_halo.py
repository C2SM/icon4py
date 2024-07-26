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
import logging
import pathlib

import mpi4py
import mpi4py.MPI
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.grid_manager as gm
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.decomposition.halo import HaloGenerator, SimpleMetisDecomposer
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

simple_distribution = xp.asarray(
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
cell_own = {0: [0, 3, 4, 6, 7, 10], 1: [1, 2, 5, 14, 17], 2: [8, 9, 11], 3: [12, 13, 15, 16]}
cell_halos = {
    0: [2, 15, 1, 11, 13, 9, 17, 5, 12, 14, 8, 16],
    1: [4, 16, 3, 8, 15, 11, 13, 0, 7, 6, 9, 10, 12],
    2: [5, 6, 12, 14, 7, 2, 1, 4, 3, 10, 15, 16, 17],
    3: [9, 10, 17, 14, 0, 1, 6, 7, 8, 2, 3, 4, 5, 11],
}


edge_own = {
    0: [1, 5, 12, 13, 14, 9],
    1: [8, 7, 6, 25, 4, 2],
    2: [16, 11, 15, 17, 10, 24],
    3: [19, 23, 22, 26, 0, 3, 20, 18, 21],
}

edge_halos = {
    0: [0, 4, 21, 10, 2, 3, 8, 6, 7, 19, 20, 17, 16, 11, 18, 26, 25, 15, 23, 24, 22],
    1: [
        5,
        12,
        22,
        23,
        3,
        1,
        9,
        15,
        16,
        11,
        19,
        20,
        0,
        17,
        24,
        21,
        26,
        13,
        10,
        14,
        18,
    ],
    2: [7, 6, 9, 8, 14, 18, 19, 23, 25, 20, 12, 13, 2, 3, 4, 5, 1, 21, 22, 0, 26],
    3: [10, 11, 13, 14, 25, 6, 24, 1, 5, 4, 8, 9, 17, 12, 15, 16, 2, 7],
}

vertex_own = {
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
vertex_halo = {
    0: [0, 1, 2, 3, 5, 6, 7, 8],
    1: [1, 2, 0, 5, 3, 8, 6, 7, 4],
    2: [8, 6, 7, 4, 0, 2, 1],
    3: [3, 4, 5],
}

owned = {dims.CellDim: cell_own, dims.EdgeDim: edge_own, dims.VertexDim: vertex_own}
halos = {dims.CellDim: cell_halos, dims.EdgeDim: edge_halos, dims.VertexDim: vertex_halo}


def test_halo_constructor_owned_cells(processor_props):  # fixture
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        ugrid=grid,
        run_properties=processor_props,
        rank_mapping=simple_distribution,
    )
    my_owned_cells = halo_generator.owned_cells()

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(cell_own[processor_props.rank])
    assert xp.setdiff1d(my_owned_cells, cell_own[processor_props.rank]).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.mpi(min_size=4)
def test_element_ownership_is_unique(dim, processor_props):  # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")
    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        ugrid=grid,
        run_properties=processor_props,
        rank_mapping=simple_distribution,
    )

    decomposition_info = halo_generator.construct_decomposition_info()
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
        # recv_buffer = xp.empty(sum(local_sizes), dtype=int)
        recv_buffer = -1 * xp.ones((4, buffer_size), dtype=int)
        print(f"{recv_buffer.shape}")
    else:
        recv_buffer = None
    # TODO (@halungge) Gatherv does not work if one of the buffers has size-0 (VertexDim)
    # comm.Gatherv(sendbuf=owned, recvbuf=(recv_buffer, local_sizes), root=0)
    comm.Gather(sendbuf=send_buf, recvbuf=recv_buffer, root=0)
    if processor_props.rank == 0:
        print(f"global indices: {recv_buffer}")
        # check there are no duplicates
        values = recv_buffer[recv_buffer != -1]
        assert values.size == len(xp.unique(values))
        # check the buffer has all global indices
        assert xp.all(xp.sort(values) == global_indices(dim))


@pytest.mark.with_mpi(min_size=4)
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_halo_constructor_decomposition_info(processor_props, dim):  # fixture
    if processor_props.comm_size != 4:
        pytest.skip("This test requires exactly 4 MPI ranks.")

    grid = simple.SimpleGrid()
    halo_generator = HaloGenerator(
        ugrid=grid,
        run_properties=processor_props,
        rank_mapping=simple_distribution,
    )

    decomp_info = halo_generator.construct_decomposition_info()
    my_halo = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.HALO)
    print(f"rank {processor_props.rank} has halo {dim} : {my_halo}")
    assert my_halo.size == len(halos[dim][processor_props.rank])
    assert xp.setdiff1d(my_halo, halos[dim][processor_props.rank], assume_unique=True).size == 0
    my_owned = decomp_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    print(f"rank {processor_props.rank} owns {dim} : {my_owned} ")
    assert my_owned.size == len(owned[dim][processor_props.rank])
    assert xp.setdiff1d(my_owned, owned[dim][processor_props.rank], assume_unique=True).size == 0


@pytest.mark.parametrize(
    "field_offset",
    [dims.C2V, dims.E2V, dims.V2C, dims.E2C, dims.C2E, dims.V2E, dims.C2E2C, dims.V2E2V],
)
def test_local_connectivities(processor_props, caplog, field_offset):  # fixture
    caplog.set_level(logging.INFO)
    grid = grid_file_manager(GRID_FILE).grid
    labels = decompose(grid, processor_props)
    halo_generator = HaloGenerator(
        ugrid=grid,
        run_properties=processor_props,
        rank_mapping=labels,
    )

    decomposition_info = halo_generator.construct_decomposition_info()

    connectivity = halo_generator.construct_local_connectivity(field_offset, decomposition_info)
    # there is an neighbor list for each index of the target dimension on the node
    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], defs.DecompositionInfo.EntryType.ALL
        ).size
    )
    # all neighbor indices are valid local indices
    assert xp.max(connectivity) == xp.max(
        decomposition_info.local_index(field_offset.source, defs.DecompositionInfo.EntryType.ALL)
    )
    # TODO what else?
    # outer halo entries have SKIP_VALUE neighbors (depends on offsets)


def grid_file_manager(file: pathlib.Path) -> gm.GridManager:
    manager = gm.GridManager(
        gm.ToGt4PyTransformation(), str(file), v_grid.VerticalGridConfig(num_levels=1)
    )
    manager()
    return manager


def global_indices(dim: dims.Dimension) -> int:
    mesh = simple.SimpleGrid()
    return xp.arange(mesh.size[dim], dtype=xp.int32)


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


@pytest.mark.xfail(reason="This test is not yet implemented")
def test_local_grid(processor_props, caplog):  # fixture
    caplog.set_level(logging.INFO)

    grid = grid_file_manager(GRID_FILE).grid
    labels = decompose(grid, processor_props)
    halo_generator = HaloGenerator(
        ugrid=grid,
        run_properties=processor_props,
        rank_mapping=labels,
    )
    decomposition_info = halo_generator.construct_decomposition_info()
    local_grid = halo_generator.local_grid(decomposition_info)

    assert (
        local_grid.num_cells
        == decomposition_info.global_index(dims.CellDim, defs.DecompositionInfo.EntryType.All).size
    )


@pytest.mark.with_mpi
def test_distributed_fields(processor_props):  # fixture
    grid_manager = grid_file_manager(GRID_FILE)

    global_grid = grid_manager.grid
    labels = decompose(global_grid, processor_props)

    halo_generator = HaloGenerator(
        ugrid=global_grid,
        run_properties=processor_props,
        rank_mapping=labels,
    )
    decomposition_info = halo_generator.construct_decomposition_info()
    # distributed read: read one field per dimension
    local_geometry_fields = grid_manager.read_geometry(decomposition_info)
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
    global_geometry_fields = grid_manager.read_geometry()
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


def decompose(grid: base_grid.BaseGrid, processor_props):
    partitioner = SimpleMetisDecomposer()
    labels = partitioner(grid.connectivities[dims.C2E2CDim], n_part=processor_props.comm_size)
    return labels


def assert_gathered_field_against_global(
    decomposition_info: defs.DecompositionInfo,
    processor_props: defs.ProcessProperties,
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
        sorted = xp.zeros(global_reference_field.shape, dtype=xp.float64)
        sorted[gathered_global_indices] = gathered_field
        assert helpers.dallclose(sorted, global_reference_field)


# TODO add test including halo access:
#  Will uses geofac_div and geofac_n2s
