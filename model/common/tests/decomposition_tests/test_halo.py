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
import numpy as np
import pytest
import xugrid as xu

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.grid_manager as gm
import icon4py.model.common.grid.vertical as v_grid
from icon4py.model.common.decomposition import definitions as defs
from icon4py.model.common.decomposition.halo import HaloGenerator
from icon4py.model.common.grid import simple
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
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


def test_halo_constructor_owned_cells(processor_props, simple_ugrid):  # fixture
    grid = simple_ugrid
    halo_generator = HaloGenerator(
        ugrid=grid,
        rank_info=processor_props,
        rank_mapping=simple_distribution,
        num_lev=1,
        face_face_connectivity=simple.SimpleGridData.c2e2c_table,
        node_face_connectivity=simple.SimpleGridData.v2c_table,
    )
    my_owned_cells = halo_generator.owned_cells()

    print(f"rank {processor_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(cell_own[processor_props.rank])
    assert xp.setdiff1d(my_owned_cells, cell_own[processor_props.rank]).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.mpi(min_size=4)
def test_element_ownership_is_unique(dim, processor_props, simple_ugrid):  # fixture
    grid = simple_ugrid
    halo_generator = HaloGenerator(
        ugrid=grid,
        rank_info=processor_props,
        rank_mapping=simple_distribution,
        num_lev=1,
        face_face_connectivity=simple.SimpleGridData.c2e2c_table,
        node_face_connectivity=simple.SimpleGridData.v2c_table,
    )
    assert processor_props.comm_size == 4, "This test requires 4 MPI ranks."

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


# TODO (@halungge) this test can be run on 4 MPI ranks or should we rather switch to a single node,
# and parametrizes the rank number
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_halo_constructor_decomposition_info(processor_props, simple_ugrid, dim):  # fixture
    grid = simple_ugrid
    halo_generator = HaloGenerator(
        ugrid=grid,
        rank_info=processor_props,
        rank_mapping=simple_distribution,
        num_lev=1,
        face_face_connectivity=simple.SimpleGridData.c2e2c_table,
        node_face_connectivity=simple.SimpleGridData.v2c_table,
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


# TODO V2E2V (from grid file vertices_of_vertex) do we use that at all?
@pytest.mark.parametrize(
    "field_offset", [dims.C2V, dims.E2V, dims.V2C, dims.E2C, dims.V2E, dims.C2E2C]
)
def test_local_connectivities(processor_props, caplog, field_offset):  # fixture
    caplog.set_level(logging.INFO)
    grid = as_ugrid2d(UGRID_FILE)
    icon_grid = grid_file_manager(GRID_FILE).grid
    distributed_grids = grid.partition(n_part=4)
    labels = grid.label_partitions(n_part=4)
    halo_generator = HaloGenerator(
        ugrid=grid,
        rank_info=processor_props,
        rank_mapping=labels,
        num_lev=1,
        face_face_connectivity=icon_grid.connectivities[dims.C2E2CDim],
        node_face_connectivity=icon_grid.connectivities[dims.V2CDim],
        node_edge_connectivity=icon_grid.connectivities[dims.V2EDim],
    )

    decomposition_info = halo_generator.construct_decomposition_info()

    connectivity = halo_generator.construct_local_connectivity(field_offset, decomposition_info)
    # TODO (@halungge): think of more valuable assertions
    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], defs.DecompositionInfo.EntryType.ALL
        ).size
    )
    assert xp.max(connectivity) == xp.max(
        decomposition_info.local_index(field_offset.source, defs.DecompositionInfo.EntryType.ALL)
    )


@pytest.fixture
def simple_ugrid() -> xu.Ugrid2d:
    """
    Programmatically construct a xugrid.ugrid.ugrid2d.Ugrid2d object

    Returns: a Ugrid2d object base on the SimpleGrid

    """
    simple_mesh = simple.SimpleGrid()
    fill_value = -1
    node_x = xp.arange(simple_mesh.num_vertices, dtype=xp.float64)
    node_y = xp.arange(simple_mesh.num_vertices, dtype=xp.float64)
    grid = xu.Ugrid2d(
        node_x,
        node_y,
        fill_value,
        projected=True,
        face_node_connectivity=simple_mesh.connectivities[dims.C2VDim],
        edge_node_connectivity=simple_mesh.connectivities[dims.E2VDim],
    )

    return grid


UGRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R_ugrid.nc"
)
GRID_FILE = dt_utils.GRIDS_PATH.joinpath(dt_utils.R02B04_GLOBAL).joinpath(
    "icon_grid_0013_R02B04_R.nc"
)


def grid_file_manager(file: pathlib.Path) -> gm.GridManager:
    manager = gm.GridManager(
        gm.ToGt4PyTransformation(), str(file), v_grid.VerticalGridConfig(num_levels=1)
    )
    manager()
    return manager
    

def as_ugrid2d(file: pathlib.Path) -> xu.Ugrid2d:
    return as_xudataset(file).grid

def as_xudataset(file: pathlib.Path) -> xu.UgridDataset:
    return xu.open_dataset(file.as_posix())

def global_indices(dim: dims.Dimension) -> int:
    mesh = simple.SimpleGrid()
    return xp.arange(mesh.size[dim], dtype=xp.int32)

def icon_distribution(props:defs.ProcessProperties, decomposition_info:defs.DecompositionInfo) -> xp.ndarray:
    cell_index =  decomposition_info.global_index(dims.CellDim, defs.DecompositionInfo.EntryType.OWNED)
    comm = props.comm
    local_sizes = xp.array(comm.gather(cell_index.size, root=0))
    if comm.rank == 0:
        recv_buffer = xp.empty(sum(local_sizes), dtype=int)
    else:
        recv_buffer = None
    comm.Gatherv(sendbuf=cell_index, recvbuf=(recv_buffer, local_sizes), root=0)
    distribution = xp.empty((sum(local_sizes)), dtype=int)
    if comm.rank == 0:
        start_index = 0
        for s in comm.size:
            end_index = local_sizes[s]
            distribution[recv_buffer[start_index:end_index]] = s
            start_index = end_index
    
    comm.Bcast(distribution, root=0)
    return distribution

@pytest.mark.xfail(reason="This test is not yet implemented")
def test_local_grid(processor_props, caplog):  # fixture
    caplog.set_level(logging.INFO)
    grid = as_ugrid2d(UGRID_FILE)
    icon_grid = grid_file_manager(GRID_FILE).grid
    distributed_grids = grid.partition(n_part=4)
    # TODO (@halungge): replace with the icon 4 nodes distribution from serialbox data.
    labels = grid.label_partitions(n_part=4)
    halo_generator = HaloGenerator(
        ugrid=grid,
        rank_info=processor_props,
        rank_mapping=labels,
        num_lev=1,
        face_face_connectivity=icon_grid.connectivities[dims.C2E2CDim],
        node_face_connectivity=icon_grid.connectivities[dims.V2CDim],
        node_edge_connectivity=icon_grid.connectivities[dims.V2EDim],
    )
    decomposition_info = halo_generator.construct_decomposition_info()
    local_grid = halo_generator.local_grid(decomposition_info)
    
    assert local_grid.num_cells == decomposition_info.global_index(dims.CellDim, defs.DecompositionInfo.EntryType.All).size
    
  
#@pytest.mark.with_mpi(min_size=4)  
def test_distributed_fields(processor_props): # fixture
    #if processor_props.comm_size != 4:
    #    pytest.skip("This test requires 4 MPI ranks.")
    grid_manager = grid_file_manager(GRID_FILE)
    processor_props.rank = 2
    ugrid = as_ugrid2d(UGRID_FILE)
    labels = ugrid.label_partitions(n_part=4)
    local_patches = ugrid.partition(n_part=4)
    halo_generator = HaloGenerator(
        ugrid=ugrid,
        rank_info=processor_props,
        rank_mapping=labels,
        num_lev=1,
        face_face_connectivity=grid_manager.grid.connectivities[dims.C2E2CDim],
        node_face_connectivity=grid_manager.grid.connectivities[dims.V2CDim],
    )
    decomposition_info = halo_generator.construct_decomposition_info()
    cell_area, edge_length = grid_manager.read_geometry(decomposition_info)
    print(f"rank = {processor_props.rank} has size(cell_area): {cell_area.shape}, has size(edge_length): {edge_length.shape}")
    assert cell_area.size == decomposition_info.global_index(dims.CellDim, defs.DecompositionInfo.EntryType.ALL).size
    assert cell_area.size <= grid_manager.grid.global_properties.num_cells
    assert edge_length.size == decomposition_info.global_index(dims.EdgeDim,
                                                             defs.DecompositionInfo.EntryType.ALL).size
    owned_cell_area = cell_area[decomposition_info.local_index(dims.CellDim, defs.DecompositionInfo.EntryType.OWNED)]
    assert np.allclose(local_patches[processor_props.rank]["cell_area"] , cell_area)
    merged = xu.merge_partitions(local_patches)

    
    
    

