# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import neighbor_sum
from icon4py.model.common.dimension import C2E2CO, C2E2CODim

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.settings import xp

@gtx.field_operator
def _nabla2_scalar(
    psi_c: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:

    nabla2_psi_c = neighbor_sum(psi_c(C2E2CO) * geofac_n2s, axis=C2E2CODim)

    return nabla2_psi_c

@gtx.program
def nabla2_scalar(
    psi_c: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    nabla2_psi_c: fa.CellField[ta.wpfloat],
):
    _nabla2_scalar(
        psi_c,
        geofac_n2s,
        out=nabla2_psi_c,
    )


def compute_smooth_topo(
    topography: fa.CellField[ta.wpfloat],
    grid: icon_grid.IconGrid,
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    num_iterations: int = 25,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed topography needed for the SLEVE coordinate.
    """

    topography_smoothed = topography.copy()

    nabla2_topo_np = xp.zeros((grid.num_cells), dtype=ta.wpfloat)
    nabla2_topo = gtx.as_field((dims.CellDim), nabla2_topo_np)

    for iter in range(num_iterations):

        nabla2_scalar(
            psi_c=topography_smoothed,
            geofac_n2s=geofac_n2s,
            z_topography=nabla2_topo,
            offset_provider={"C2E2CO":grid.get_offset_provider("C2E2CO"),}
        )
        
        topography_smoothed_np = topography_smoothed.asnumpy() + 0.125 * nabla2_topo.asnumpy() * grid.cell_areas # TODO implement this
        topography_smoothed = gtx.as_field((dims.CellDim), topography_smoothed_np)

    return topography_smoothed