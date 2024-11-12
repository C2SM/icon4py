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
        math_oper.nabla2_scalar_2D.with_backend(backend)(
            psi_c=topography_smoothed,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_topo,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            offset_provider={
                "C2E2CO": grid.get_offset_provider("C2E2CO"),
            },
        )

        topography_smoothed_np = (
            topography_smoothed.asnumpy() + 0.125 * nabla2_topo.asnumpy() * cell_areas.asnumpy()
        )
        topography_smoothed = gtx.as_field((dims.CellDim,), topography_smoothed_np)

    return gtx.as_field((dims.CellDim,), topography_smoothed_np)
