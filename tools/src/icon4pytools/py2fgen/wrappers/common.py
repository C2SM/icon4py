# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# type: ignore

import logging

import numpy as np
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import base, horizontal, icon


log = logging.getLogger(__name__)


def adjust_fortran_indices(inp: xp.ndarray, offset: int) -> xp.ndarray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return xp.subtract(inp, offset)


def construct_icon_grid(
    cell_starts: np.ndarray,
    cell_ends: xp.ndarray,
    vertex_starts: np.ndarray,
    vertex_ends: np.ndarray,
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    c2e: np.ndarray,
    e2c: np.ndarray,
    c2e2c: np.ndarray,
    e2c2e: np.ndarray,
    e2v: np.ndarray,
    v2e: np.ndarray,
    v2c: np.ndarray,
    e2c2v: np.ndarray,
    c2v: np.ndarray,
    grid_id: str,
    global_grid_params: icon.GlobalGridParams,
    num_vertices: int,
    num_cells: int,
    num_edges: int,
    vertical_size: int,
    limited_area: bool,
    on_gpu: bool,
):
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivitity arrays by 1")
    offset = 1

    cells_start_index = adjust_fortran_indices(cell_starts, offset)
    vertex_start_index = adjust_fortran_indices(vertex_starts, offset)
    edge_start_index = adjust_fortran_indices(edge_starts, offset)

    cells_end_index = cell_ends.ndarray
    vertex_end_index = vertex_ends.ndarray
    edge_end_index = edge_ends.ndarray

    c2e = adjust_fortran_indices(c2e, offset)
    c2v = adjust_fortran_indices(c2v, offset)
    v2c = adjust_fortran_indices(v2c, offset)
    e2v = adjust_fortran_indices(e2v, offset)[
        :, 0:2
    ]  # slicing required for e2v as input data is actually e2c2v
    c2e2c = adjust_fortran_indices(c2e2c, offset)
    v2e = adjust_fortran_indices(v2e, offset)
    e2c2v = adjust_fortran_indices(e2c2v, offset)
    e2c = adjust_fortran_indices(e2c, offset)
    e2c2e = adjust_fortran_indices(e2c2e, offset)

    # stacked arrays
    c2e2c0 = np.column_stack((np.asarray(range(c2e2c.shape[0])), c2e2c.ndarray))
    e2c2e0 = np.column_stack((np.asarray(range(e2c2e.shape[0])), e2c2e.ndarray))

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

    grid = (
        icon.IconGrid(id_=grid_id)
        .with_config(config)
        .with_global_params(global_grid_params)
        .with_start_end_indices(dims.VertexDim, vertex_start_index, vertex_end_index)
        .with_start_end_indices(dims.EdgeDim, edge_start_index, edge_end_index)
        .with_start_end_indices(dims.CellDim, cells_start_index, cells_end_index)
        .with_connectivities(
            {
                dims.C2EDim: c2e,
                dims.C2VDim: c2v,
                dims.E2CDim: e2c,
                dims.E2C2EDim: e2c2e,
                dims.C2E2CDim: c2e2c,
                dims.C2E2CODim: c2e2c0,
                dims.E2C2EODim: e2c2e0,
            }
        )
        .with_connectivities(
            {
                dims.V2EDim: v2e,
                dims.E2VDim: e2v,
                dims.E2C2VDim: e2c2v,
                dims.V2CDim: v2c,
            }
        )
    )

    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
        }
    )

    return grid


def construct_decomposition(
    c_glb_index: np.ndarray,
    e_glb_index: np.ndarray,
    v_glb_index: np.ndarray,
    c_owner_mask: np.ndarray,
    e_owner_mask: np.ndarray,
    v_owner_mask: np.ndarray,
    num_cells: int,
    num_edges: int,
    num_vertices: int,
    num_levels: int,
    comm_id: int,
) -> tuple[
    definitions.ProcessProperties, definitions.DecompositionInfo, definitions.ExchangeRuntime
]:
    log.debug("Offsetting Fortran connectivitity arrays by 1")
    offset = 1

    c_glb_index = adjust_fortran_indices(c_glb_index, offset)
    e_glb_index = adjust_fortran_indices(e_glb_index, offset)
    v_glb_index = adjust_fortran_indices(v_glb_index, offset)

    c_owner_mask = c_owner_mask[:num_cells]
    e_owner_mask = e_owner_mask[:num_edges]
    v_owner_mask = v_owner_mask[:num_vertices]

    decomposition_info = (
        definitions.DecompositionInfo(
            klevels=num_levels, num_cells=num_cells, num_edges=num_edges, num_vertices=num_vertices
        )
        .with_dimension(dims.CellDim, c_glb_index, c_owner_mask)
        .with_dimension(dims.EdgeDim, e_glb_index, e_owner_mask)
        .with_dimension(dims.VertexDim, v_glb_index, v_owner_mask)
    )
    processor_props = definitions.get_processor_properties(definitions.MultiNodeRun(), comm_id)
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    return processor_props, decomposition_info, exchange
