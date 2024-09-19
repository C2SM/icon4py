# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# type: ignore

import cProfile
import logging
import pstats

from gt4py.next import Dimension
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32
from icon4py.model.atmosphere.diffusion.diffusion import Diffusion
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.grid import base, horizontal, icon
from icon4py.model.common.grid.icon import GlobalGridParams
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)

GLOBAL_STATE = {"diffusion_granule": Diffusion(), "icon_grid": None, "profiler": cProfile.Profile()}

# extra dimensions
CellIndexDim = Dimension("CellIndex")
EdgeIndexDim = Dimension("EdgeIndex")
VertexIndexDim = Dimension("VertexIndex")


# profiling utils
def profile_enable():
    GLOBAL_STATE["profiler"].enable()


def profile_disable():
    GLOBAL_STATE["profiler"].disable()
    stats = pstats.Stats(GLOBAL_STATE["profiler"])
    stats.dump_stats(f"{__name__}.profile")


# initialising the grid
def grid_init(
    grid_id: str,
    cell_starts: Field[[CellIndexDim], int32],
    cell_ends: Field[[CellIndexDim], int32],
    vertex_starts: Field[[VertexIndexDim], int32],
    vertex_ends: Field[[VertexIndexDim], int32],
    edge_starts: Field[[EdgeIndexDim], int32],
    edge_ends: Field[[EdgeIndexDim], int32],
    c2e: Field[[dims.CellDim, dims.C2EDim], int32],
    e2c: Field[[dims.EdgeDim, dims.E2CDim], int32],
    c2e2c: Field[[dims.CellDim, dims.C2E2CDim], int32],
    e2c2e: Field[[dims.EdgeDim, dims.E2C2EDim], int32],
    e2v: Field[[dims.EdgeDim, dims.E2VDim], int32],
    v2e: Field[[dims.VertexDim, dims.V2EDim], int32],
    v2c: Field[[dims.VertexDim, dims.V2CDim], int32],
    e2c2v: Field[[dims.EdgeDim, dims.E2C2VDim], int32],
    c2v: Field[[dims.CellDim, dims.C2VDim], int32],
    c2e2c2e: Field[[dims.CellDim, dims.C2E2C2EDim], int32],
    global_root: int32,
    global_level: int32,
    num_vertices: int32,
    num_cells: int32,
    num_edges: int32,
    vertical_size: int32,
    limited_area: bool,
):
    global_grid_params = GlobalGridParams(level=global_level, root=global_root)

    GLOBAL_STATE["icon_grid"] = construct_icon_grid(
        grid_id=grid_id,
        global_grid_params=global_grid_params,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=True if settings.device == "GPU" else False,
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        c2e2c2e=c2e2c2e,
    )


def offset_fortran_indices_return_numpy(inp) -> xp.ndarray:
    # todo: maybe needed in Fortran? (breaks datatest)
    #   return xp.subtract(inp.ndarray, 1)  # noqa: ERA001
    return inp.ndarray


def offset_squeeze_fortran_indices_return_xp(inp) -> xp.ndarray:
    # todo: maybe needed in Fortran? (breaks datatest)
    #   return xp.squeeze(xp.subtract(inp.ndarray, 1))  # noqa: ERA001
    #   might only be needed for Fortran
    return inp.ndarray


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
    e2c2e,
    e2v,
    v2e,
    v2c,
    e2c2v,
    c2v,
    c2e2c2e,
):
    num_levels = vertical_size
    log.debug("Constructing icon grid in py")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", num_levels)

    cells_start_index_np = offset_fortran_indices_return_numpy(cell_starts)
    vertex_start_index_np = offset_fortran_indices_return_numpy(vertex_starts)
    edge_start_index_np = offset_fortran_indices_return_numpy(edge_starts)

    cells_end_index_np = cell_ends.ndarray
    vertex_end_index_np = vertex_ends.ndarray
    edge_end_index_np = edge_ends.ndarray

    c2e_loc = offset_squeeze_fortran_indices_return_xp(c2e)
    c2v_loc = offset_squeeze_fortran_indices_return_xp(c2v)
    v2c_loc = offset_squeeze_fortran_indices_return_xp(v2c)
    e2v_loc = offset_squeeze_fortran_indices_return_xp(e2v)
    c2e2c_loc = offset_squeeze_fortran_indices_return_xp(c2e2c)
    c2e2c2e_loc = offset_squeeze_fortran_indices_return_xp(c2e2c2e)
    v2e_loc = offset_squeeze_fortran_indices_return_xp(v2e)
    e2c2v_loc = offset_squeeze_fortran_indices_return_xp(e2c2v)
    e2c_loc = offset_squeeze_fortran_indices_return_xp(e2c)
    e2c2e_loc = offset_squeeze_fortran_indices_return_xp(e2c2e)

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
    log.debug(" c2e2c.shape[0] %s", c2e2c_loc.shape[0])
    log.debug(" xp.asarray(range(c2e2c.shape[0]))) %s", xp.asarray(range(c2e2c_loc.shape[0])).shape)
    c2e2c0 = xp.column_stack((xp.asarray(range(c2e2c_loc.shape[0])), c2e2c_loc))

    e2c2e0 = xp.column_stack((xp.asarray(range(e2c2e_loc.shape[0])), e2c2e_loc))

    grid = (
        icon.IconGrid(id_=grid_id)
        .with_config(config)
        .with_global_params(global_grid_params)
        .with_start_end_indices(dims.VertexDim, vertex_start_index_np, vertex_end_index_np)
        .with_start_end_indices(dims.EdgeDim, edge_start_index_np, edge_end_index_np)
        .with_start_end_indices(dims.CellDim, cells_start_index_np, cells_end_index_np)
        .with_connectivities(
            {
                dims.C2EDim: c2e_loc,
                dims.C2VDim: c2v_loc,
                dims.E2CDim: e2c_loc,
                dims.E2C2EDim: e2c2e_loc,
                dims.C2E2CDim: c2e2c_loc,
                dims.C2E2CODim: c2e2c0,
                dims.E2C2EODim: e2c2e0,
                dims.C2E2C2EDim: c2e2c2e_loc,
            }
        )
        .with_connectivities(
            {
                dims.V2EDim: v2e_loc,
                dims.E2VDim: e2v_loc,
                dims.E2C2VDim: e2c2v_loc,
                dims.V2CDim: v2c_loc,
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
