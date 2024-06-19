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
from icon4py.model.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CEDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    ECVDim,
    EdgeDim,
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
    log.debug(" c2e2c.shape[0] %s", c2e2c.shape[0])
    log.debug(" xp.asarray(range(c2e2c.shape[0]))) %s", xp.asarray(range(c2e2c.shape[0])).shape)
    c2e2c0 = xp.column_stack(((xp.asarray(range(c2e2c.shape[0]))), c2e2c))

    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_start_index, vertex_end_index)
        .with_start_end_indices(EdgeDim, edge_start_index, edge_end_index)
        .with_start_end_indices(CellDim, cells_start_index, cells_end_index)
        .with_connectivities(
            {
                C2EDim: c2e,
                E2CDim: e2c,
                C2E2CDim: c2e2c,
                C2E2CODim: c2e2c0,
            }
        )
        .with_connectivities(
            {
                V2EDim: v2e,
                E2C2VDim: e2c2v,
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


def fortran_grid_indices_to_numpy_offset(inp) -> np.ndarray:
    return np.subtract(inp.asnumpy(), 1)


def fortran_grid_connectivities_to_xp_offset(inp) -> xp.ndarray:
    return xp.squeeze(xp.subtract(inp.ndarray, 1))


def fortran_grid_indices_to_numpy(inp) -> np.ndarray:
    return inp.asnumpy()
