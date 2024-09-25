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

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import SolveNonhydro
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal, icon
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)

GLOBAL_STATE = {
    "diffusion_granule": Diffusion(),
    "profiler": cProfile.Profile(),
    "dycore_granule": SolveNonhydro(),
}


# profiling utils
def profile_enable():
    GLOBAL_STATE["profiler"].enable()


def profile_disable():
    GLOBAL_STATE["profiler"].disable()
    stats = pstats.Stats(GLOBAL_STATE["profiler"])
    stats.dump_stats(f"{__name__}.profile")


def adjust_fortran_indices(inp: xp.ndarray, offset: int) -> xp.ndarray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return xp.subtract(inp.ndarray, offset)


def construct_icon_grid(
    c2e,
    e2c,
    c2e2c,
    e2c2e,
    e2v,
    v2e,
    v2c,
    e2c2v,
    c2v,
    cell_starts,
    cell_ends,
    vertex_starts,
    vertex_ends,
    edge_starts,
    edge_ends,
    global_grid_params,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
    on_gpu,
    grid_id,
):
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivity arrays")

    # We only need to subtract 1 from start index arrays
    cell_starts = adjust_fortran_indices(cell_starts, offset=1)
    cell_ends = adjust_fortran_indices(cell_ends, offset=0)
    vertex_starts = adjust_fortran_indices(vertex_starts, offset=1)
    vertex_ends = adjust_fortran_indices(vertex_ends, offset=0)
    edge_starts = adjust_fortran_indices(edge_starts, offset=1)
    edge_ends = adjust_fortran_indices(edge_ends, offset=0)

    # Offset indices
    c2e = adjust_fortran_indices(c2e, offset=1)
    e2c = adjust_fortran_indices(e2c, offset=1)
    c2e2c = adjust_fortran_indices(c2e2c, offset=1)
    e2c2e = adjust_fortran_indices(e2c2e, offset=1)
    e2v = adjust_fortran_indices(e2v, offset=1)[
        :, 0:2
    ]  # todo(samkellerhals): Find out if slicing is required for Fortran inputs
    v2e = adjust_fortran_indices(v2e, offset=1)
    v2c = adjust_fortran_indices(v2c, offset=1)
    e2c2v = adjust_fortran_indices(e2c2v, offset=1)
    c2v = adjust_fortran_indices(c2v, offset=1)

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
    c2e2c = c2e2c
    e2c2e = e2c2e
    c2e2c0 = xp.column_stack(((range(c2e2c.shape[0])), c2e2c))
    e2c2e0 = xp.column_stack(((range(e2c2e.shape[0])), e2c2e))
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

    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
        }
    )

    return grid
