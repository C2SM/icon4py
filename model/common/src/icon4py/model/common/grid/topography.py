# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, neighbor_sum
from icon4py.model.common.dimension import C2E2CO, C2E2CODim

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.settings import xp

@gtx.field_operator
def _nabla2_scalar(
    psi_c: fa.CellKField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres.
    """
    nabla2_psi_c = neighbor_sum(psi_c(C2E2CO) * geofac_n2s, axis=C2E2CODim)

    return nabla2_psi_c

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def nabla2_scalar(
    psi_c: fa.CellKField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    nabla2_psi_c: fa.CellKField[ta.wpfloat],
):
    _nabla2_scalar(
        psi_c,
        geofac_n2s,
        out=nabla2_psi_c,
    )


def compute_smooth_topo(
    topography: fa.CellField[ta.wpfloat],
    grid: icon_grid.IconGrid,
    cell_areas: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    num_iterations: int = 25,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    #topography_smoothed = gtx.as_field((dims.CellDim,), topography.asnumpy())
    topography_smoothed_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=ta.wpfloat)
    topography_smoothed_np[:,0] = topography.asnumpy()
    topography_smoothed = gtx.as_field((dims.CellDim, dims.KDim), topography_smoothed_np)

    nabla2_topo_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=ta.wpfloat)
    nabla2_topo = gtx.as_field((dims.CellDim, dims.KDim), nabla2_topo_np)

    for iter in range(num_iterations):

        nabla2_scalar(
            psi_c=topography_smoothed,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_topo,
            offset_provider={"C2E2CO":grid.get_offset_provider("C2E2CO"),}
        )
        
        topography_smoothed_np[:,0] = topography_smoothed.asnumpy()[:,0] + 0.125 * nabla2_topo.asnumpy()[:,0] * cell_areas.asnumpy()
        topography_smoothed = gtx.as_field((dims.CellDim, dims.KDim), topography_smoothed_np)

    return topography_smoothed