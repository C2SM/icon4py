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

import numpy as np
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    MultiNodeRun,
)
from icon4py.model.common.decomposition.mpi_decomposition import get_multinode_properties
from icon4py.model.common.dimension import (
    C2E2C2EDim,
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    C2VDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    ECDim,
    ECVDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.base import GridConfig, HorizontalGridSize, VerticalGridSize
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)


def construct_icon_grid(
    cells_start_index,
    cells_end_index,
    vertex_start_index,
    vertex_end_index,
    edge_start_index,
    edge_end_index,
    num_cells,
    num_edges,
    num_vertices,
    num_levels,
    c2e,
    c2e2c,
    v2e,
    e2c2v,
    e2c,
    limited_area: bool,
    on_gpu: bool,
) -> IconGrid:
    log.debug("Constructing icon grid in py")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", num_levels)

    cells_start_index_np = offset_fortran_indices_return_numpy(cells_start_index)
    vertex_start_index_np = offset_fortran_indices_return_numpy(vertex_start_index)
    edge_start_index_np = offset_fortran_indices_return_numpy(edge_start_index)

    cells_end_index_np = cells_end_index.asnumpy()
    vertex_end_index_np = vertex_end_index.asnumpy()
    edge_end_index_np = edge_end_index.asnumpy()

    c2e_loc = offset_squeeze_fortran_indices_return_xp(c2e)
    c2e2c_loc = offset_squeeze_fortran_indices_return_xp(c2e2c)
    v2e_loc = offset_squeeze_fortran_indices_return_xp(v2e)
    e2c2v_loc = offset_squeeze_fortran_indices_return_xp(e2c2v)
    e2c_loc = offset_squeeze_fortran_indices_return_xp(e2c)

    config = GridConfig(
        horizontal_config=HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_config=VerticalGridSize(num_lev=num_levels),
        limited_area=limited_area,
        on_gpu=on_gpu,
    )
    log.debug(" c2e2c.shape[0] %s", c2e2c_loc.shape[0])
    log.debug(" xp.asarray(range(c2e2c.shape[0]))) %s", xp.asarray(range(c2e2c_loc.shape[0])).shape)
    c2e2c0 = xp.column_stack(((xp.asarray(range(c2e2c_loc.shape[0]))), c2e2c_loc))

    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_start_index_np, vertex_end_index_np)
        .with_start_end_indices(EdgeDim, edge_start_index_np, edge_end_index_np)
        .with_start_end_indices(CellDim, cells_start_index_np, cells_end_index_np)
        .with_connectivities(
            {
                C2EDim: c2e_loc,
                E2CDim: e2c_loc,
                C2E2CDim: c2e2c_loc,
                C2E2CODim: c2e2c0,
            }
        )
        .with_connectivities(
            {
                V2EDim: v2e_loc,
                E2C2VDim: e2c2v_loc,
            }
        )
    )

    grid.update_size_connectivities(
        {
            ECVDim: grid.size[EdgeDim] * grid.size[E2C2VDim],
            CEDim: grid.size[CellDim] * grid.size[C2EDim],
            CECDim: grid.size[CellDim] * grid.size[C2E2CDim],
        }
    )

    return grid


def construct_icon_grid_solve_nh(
    cells_start_index,
    cells_end_index,
    vertex_start_index,
    vertex_end_index,
    edge_start_index,
    edge_end_index,
    num_cells,
    num_edges,
    num_vertices,
    num_levels,
    c2e,
    c2e2c,
    v2e,
    e2c2v,
    e2c,
    e2c2e,
    e2v,
    v2c,
    c2v,
    limited_area: bool,
    on_gpu: bool,
) -> IconGrid:
    log.debug("Constructing icon grid in py")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", num_levels)

    cells_start_index_np = offset_fortran_indices_return_numpy(cells_start_index)
    vertex_start_index_np = offset_fortran_indices_return_numpy(vertex_start_index)
    edge_start_index_np = offset_fortran_indices_return_numpy(edge_start_index)

    cells_end_index_np = cells_end_index.asnumpy()
    vertex_end_index_np = vertex_end_index.asnumpy()
    edge_end_index_np = edge_end_index.asnumpy()

    c2e_loc = offset_squeeze_fortran_indices_return_xp(c2e)
    c2e2c_loc = offset_squeeze_fortran_indices_return_xp(c2e2c)
    v2e_loc = offset_squeeze_fortran_indices_return_xp(v2e)
    e2c2v_loc = offset_squeeze_fortran_indices_return_xp(e2c2v)
    e2c_loc = offset_squeeze_fortran_indices_return_xp(e2c)
    e2c2e_loc = offset_squeeze_fortran_indices_return_xp(e2c2e)
    e2v_loc = offset_squeeze_fortran_indices_return_xp(e2v)
    v2c_loc = offset_squeeze_fortran_indices_return_xp(v2c)
    c2v_loc = offset_squeeze_fortran_indices_return_xp(c2v)

    config = GridConfig(
        horizontal_config=HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_config=VerticalGridSize(num_lev=num_levels),
        limited_area=limited_area,
        on_gpu=on_gpu,
    )
    log.debug(" c2e2c.shape[0] %s", c2e2c_loc.shape[0])
    log.debug(" xp.asarray(range(c2e2c.shape[0]))) %s", xp.asarray(range(c2e2c_loc.shape[0])).shape)
    c2e2c0 = xp.column_stack(((xp.asarray(range(c2e2c_loc.shape[0]))), c2e2c_loc))
    e2c2e0 = np.column_stack(((np.asarray(range(e2c2e_loc.shape[0]))), e2c2e_loc))
    c2e2c2e = np.zeros((num_cells, 9), dtype=int)

    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_start_index_np, vertex_end_index_np)
        .with_start_end_indices(EdgeDim, edge_start_index_np, edge_end_index_np)
        .with_start_end_indices(CellDim, cells_start_index_np, cells_end_index_np)
        .with_connectivities(
            {
                C2EDim: c2e_loc,
                E2CDim: e2c_loc,
                C2E2CDim: c2e2c_loc,
                C2E2CODim: c2e2c0,
                C2E2C2EDim: c2e2c2e,
                E2C2EDim: e2c2e_loc,
                E2C2EODim: e2c2e0,
            }
        )
        .with_connectivities(
            {
                E2VDim: e2v_loc,
                V2EDim: v2e_loc,
                V2CDim: v2c_loc,
                E2C2VDim: e2c2v_loc,
                C2VDim: c2v_loc,
            }
        )
    )

    grid.update_size_connectivities(
        {
            ECVDim: grid.size[EdgeDim] * grid.size[E2C2VDim],
            CEDim: grid.size[CellDim] * grid.size[C2EDim],
            CECDim: grid.size[CellDim] * grid.size[C2E2CDim],
            ECDim: grid.size[EdgeDim] * grid.size[E2CDim],
        }
    )

    return grid


def construct_decomposition(
    c_glb_index,
    e_glb_index,
    v_glb_index,
    c_owner_mask,
    e_owner_mask,
    v_owner_mask,
    num_cells: int,
    num_edges: int,
    num_verts: int,
    num_levels: int,
    comm_id: int,
):
    c_glb_index_np = offset_fortran_indices_return_numpy(c_glb_index)
    e_glb_index_np = offset_fortran_indices_return_numpy(e_glb_index)
    v_glb_index_np = offset_fortran_indices_return_numpy(v_glb_index)

    c_owner_mask_np = c_owner_mask.asnumpy()[0:num_cells]
    e_owner_mask_np = e_owner_mask.asnumpy()[0:num_edges]
    v_owner_mask_np = v_owner_mask.asnumpy()[0:num_verts]

    decomposition_info = (
        DecompositionInfo(klevels=num_levels)
        .with_dimension(CellDim, c_glb_index_np, c_owner_mask_np)
        .with_dimension(EdgeDim, e_glb_index_np, e_owner_mask_np)
        .with_dimension(VertexDim, v_glb_index_np, v_owner_mask_np)
    )
    processor_props = get_multinode_properties(MultiNodeRun(), comm_id)
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    return processor_props, decomposition_info, exchange


def offset_fortran_indices_return_numpy(inp) -> np.ndarray:
    return np.subtract(inp.asnumpy(), 1)


def offset_squeeze_fortran_indices_return_xp(inp) -> xp.ndarray:
    return xp.squeeze(xp.subtract(inp.ndarray, 1))
