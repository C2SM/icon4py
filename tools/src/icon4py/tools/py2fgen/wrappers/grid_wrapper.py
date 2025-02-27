# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore

from typing import Optional

from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.tools.py2fgen import settings
from icon4py.tools.py2fgen.wrappers import (
    common as wrapper_common,
    debug_utils as wrapper_debug_utils,
)
from icon4py.tools.py2fgen.wrappers.wrapper_dimension import (
    CellGlobalIndexDim,
    CellIndexDim,
    EdgeGlobalIndexDim,
    EdgeIndexDim,
    VertexGlobalIndexDim,
    VertexIndexDim,
)


# TODO(havogt): remove module global state
grid: Optional[icon_grid.IconGrid] = None
exchange_runtime: Optional[decomposition_defs.ExchangeRuntime] = None


def grid_init(
    cell_starts: gtx.Field[gtx.Dims[CellIndexDim], gtx.int32],
    cell_ends: gtx.Field[gtx.Dims[CellIndexDim], gtx.int32],
    vertex_starts: gtx.Field[gtx.Dims[VertexIndexDim], gtx.int32],
    vertex_ends: gtx.Field[gtx.Dims[VertexIndexDim], gtx.int32],
    edge_starts: gtx.Field[gtx.Dims[EdgeIndexDim], gtx.int32],
    edge_ends: gtx.Field[gtx.Dims[EdgeIndexDim], gtx.int32],
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    e2c: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    c2e2c: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.int32],
    e2c2e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], gtx.int32],
    e2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2VDim], gtx.int32],
    v2e: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.int32],
    v2c: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.int32],
    e2c2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.int32],
    c2v: gtx.Field[gtx.Dims[dims.CellDim, dims.C2VDim], gtx.int32],
    c_owner_mask: gtx.Field[[dims.CellDim], bool],
    e_owner_mask: gtx.Field[[dims.EdgeDim], bool],
    v_owner_mask: gtx.Field[[dims.VertexDim], bool],
    c_glb_index: gtx.Field[[CellGlobalIndexDim], gtx.int32],
    e_glb_index: gtx.Field[[EdgeGlobalIndexDim], gtx.int32],
    v_glb_index: gtx.Field[[VertexGlobalIndexDim], gtx.int32],
    comm_id: gtx.int32,
    global_root: gtx.int32,
    global_level: gtx.int32,
    num_vertices: gtx.int32,
    num_cells: gtx.int32,
    num_edges: gtx.int32,
    vertical_size: gtx.int32,
    limited_area: bool,
) -> None:
    on_gpu = settings.config.device == settings.Device.GPU
    xp = c2e.array_ns

    # TODO(havogt): add direct support for ndarrays in py2fgen
    cell_starts = cell_starts.ndarray
    cell_ends = cell_ends.ndarray
    vertex_starts = vertex_starts.ndarray
    vertex_ends = vertex_ends.ndarray
    edge_starts = edge_starts.ndarray
    edge_ends = edge_ends.ndarray
    c_owner_mask = c_owner_mask.ndarray if c_owner_mask is not None else None
    e_owner_mask = e_owner_mask.ndarray if e_owner_mask is not None else None
    v_owner_mask = v_owner_mask.ndarray if v_owner_mask is not None else None
    c_glb_index = c_glb_index.ndarray if c_glb_index is not None else None
    e_glb_index = e_glb_index.ndarray if e_glb_index is not None else None
    v_glb_index = v_glb_index.ndarray if v_glb_index is not None else None

    if on_gpu:
        cp = xp
        cell_starts = cp.asnumpy(cell_starts)
        cell_ends = cp.asnumpy(cell_ends)
        vertex_starts = cp.asnumpy(vertex_starts)
        vertex_ends = cp.asnumpy(vertex_ends)
        edge_starts = cp.asnumpy(edge_starts)
        edge_ends = cp.asnumpy(edge_ends)
        c_owner_mask = cp.asnumpy(c_owner_mask) if c_owner_mask is not None else None
        e_owner_mask = cp.asnumpy(e_owner_mask) if e_owner_mask is not None else None
        v_owner_mask = cp.asnumpy(v_owner_mask) if v_owner_mask is not None else None
        c_glb_index = cp.asnumpy(c_glb_index) if c_glb_index is not None else None
        e_glb_index = cp.asnumpy(e_glb_index) if e_glb_index is not None else None
        v_glb_index = cp.asnumpy(v_glb_index) if v_glb_index is not None else None

    global_grid_params = icon_grid.GlobalGridParams(level=global_level, root=global_root)

    global grid
    grid = wrapper_common.construct_icon_grid(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e.ndarray,
        e2c=e2c.ndarray,
        c2e2c=c2e2c.ndarray,
        e2c2e=e2c2e.ndarray,
        e2v=e2v.ndarray,
        v2e=v2e.ndarray,
        v2c=v2c.ndarray,
        e2c2v=e2c2v.ndarray,
        c2v=c2v.ndarray,
        grid_id="icon_grid",
        global_grid_params=global_grid_params,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=on_gpu,
    )

    global exchange_runtime
    if settings.config.parallel_run:
        # Set MultiNodeExchange as exchange runtime
        (
            processor_props,
            decomposition_info,
            _exchange_runtime,
        ) = wrapper_common.construct_decomposition(
            c_glb_index,
            e_glb_index,
            v_glb_index,
            c_owner_mask,
            e_owner_mask,
            v_owner_mask,
            num_cells,
            num_edges,
            num_vertices,
            vertical_size,
            comm_id,
        )
        wrapper_debug_utils.print_grid_decomp_info(
            grid,
            processor_props,
            decomposition_info,
            num_cells,
            num_edges,
            num_vertices,
        )
        exchange_runtime = _exchange_runtime
    else:
        # set exchange runtime to SingleNodeExchange
        exchange_runtime = decomposition_defs.SingleNodeExchange()
