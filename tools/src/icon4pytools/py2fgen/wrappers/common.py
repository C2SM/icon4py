# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal, icon
from icon4py.model.common.settings import xp


def construct_icon_grid(
    grid_id,
    global_grid_params,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
    on_gpu,
    cell_starts,
    cell_ends,
    vertex_starts,
    vertex_ends,
    edge_starts,
    edge_ends,
    c2e,
    e2c,
    c2e2c,
    c2e2c2e,
    e2c2e,
    e2v,
    v2e,
    v2c,
    e2c2v,
    c2v,
):
    # Creating GridConfig instance
    config = base.GridConfig(
        horizontal_config=horizontal.HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=on_gpu,
    )

    # Prepare connectivity matrices with extra columns
    c2e2c0 = xp.column_stack((range(c2e2c.shape[0]), c2e2c))
    e2c2e0 = xp.column_stack((range(e2c2e.shape[0]), e2c2e))

    # Construct the IconGrid instance
    grid = (
        icon.IconGrid(grid_id)
        .with_config(config)
        .with_global_params(global_grid_params)
        .with_start_end_indices(dims.VertexDim, vertex_starts, vertex_ends)
        .with_start_end_indices(dims.EdgeDim, edge_starts, edge_ends)
        .with_start_end_indices(dims.CellDim, cell_starts, cell_ends)
        .with_connectivities(
            {
                dims.C2EDim: c2e,
                dims.E2CDim: e2c,
                dims.C2E2CDim: c2e2c,
                dims.C2E2CODim: c2e2c0,
                dims.C2E2C2EDim: c2e2c2e,
                dims.E2C2EDim: e2c2e,
                dims.E2C2EODim: e2c2e0,
            }
        )
        .with_connectivities(
            {
                dims.E2VDim: e2v,
                dims.V2EDim: v2e,
                dims.V2CDim: v2c,
                dims.E2C2VDim: e2c2v,
                dims.C2VDim: c2v,
            }
        )
    )

    # Update size connectivities
    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
        }
    )

    return grid
