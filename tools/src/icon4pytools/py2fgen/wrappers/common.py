# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# type: ignore

import logging

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import base, horizontal, icon
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)


def adjust_fortran_indices(inp: xp.ndarray, offset: int) -> xp.ndarray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return xp.subtract(inp.ndarray, offset)


def construct_icon_grid(
    cell_starts: xp.ndarray,
    cell_ends: xp.ndarray,
    vertex_starts: xp.ndarray,
    vertex_ends: xp.ndarray,
    edge_starts: xp.ndarray,
    edge_ends: xp.ndarray,
    c2e: xp.ndarray,
    e2c: xp.ndarray,
    c2e2c: xp.ndarray,
    e2c2e: xp.ndarray,
    e2v: xp.ndarray,
    v2e: xp.ndarray,
    v2c: xp.ndarray,
    e2c2v: xp.ndarray,
    c2v: xp.ndarray,
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

    cells_end_index = cell_ends
    vertex_end_index = vertex_ends
    edge_end_index = edge_ends

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
    c2e2c0 = xp.column_stack((xp.asarray(range(c2e2c.shape[0])), xp.asarray(c2e2c.asnumpy())))
    e2c2e0 = xp.column_stack((xp.asarray(range(e2c2e.shape[0])), xp.asarray(e2c2e.asnumpy())))

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
    c_glb_index: xp.ndarray,
    e_glb_index: xp.ndarray,
    v_glb_index: xp.ndarray,
    c_owner_mask: xp.ndarray,
    e_owner_mask: xp.ndarray,
    v_owner_mask: xp.ndarray,
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

    c_owner_mask = c_owner_mask[:num_cells] if c_owner_mask is not None else c_owner_mask
    e_owner_mask = e_owner_mask[:num_edges] if e_owner_mask is not None else e_owner_mask
    v_owner_mask = v_owner_mask[:num_vertices] if v_owner_mask is not None else v_owner_mask

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
