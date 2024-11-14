# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.math import operators as math_oper
from icon4py.model.common.settings import xp


@gtx.field_operator
def _update_topo_smooth(
    topography_smoothed: fa.CellField[ta.wpfloat],
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:
    """
    Updates the smoothed topography field inside the loop.
    """
    return topography_smoothed + 0.125 * nabla2_topo * cell_areas


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_topo_smooth(
    topography_smoothed: fa.CellField[ta.wpfloat],
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _update_topo_smooth(
        topography_smoothed=topography_smoothed,
        nabla2_topo=nabla2_topo,
        cell_areas=cell_areas,
        out=topography_smoothed,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )


def compute_smooth_topo(
    topography: fa.CellField[ta.wpfloat],
    grid: icon_grid.IconGrid,
    cell_areas: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    backend,
    num_iterations: int = 25,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    topography_smoothed = gtx.as_field((dims.CellDim,), topography.ndarray)

    nabla2_topo_np = xp.zeros((grid.num_cells,), dtype=ta.wpfloat)
    nabla2_topo = gtx.as_field((dims.CellDim,), nabla2_topo_np)

    for _ in range(num_iterations):
        math_oper.compute_nabla2_on_cell.with_backend(backend)(
            psi_c=topography_smoothed,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_topo,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            offset_provider={
                "C2E2CO": grid.get_offset_provider("C2E2CO"),
            },
        )

        update_topo_smooth.with_backend(backend)(
            topography_smoothed=topography_smoothed,
            nabla2_topo=nabla2_topo,
            cell_areas=cell_areas,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            offset_provider={},
        )

    return topography_smoothed
