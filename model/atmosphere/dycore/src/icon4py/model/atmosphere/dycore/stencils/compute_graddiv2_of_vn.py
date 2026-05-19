# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2EO, E2C2EODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_graddiv2_of_vn(
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], wpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_25."""
    z_graddiv_vn_wp = astype(horizontal_gradient_of_normal_wind_divergence, wpfloat)

    z_graddiv2_vn_wp = neighbor_sum(z_graddiv_vn_wp(E2C2EO) * geofac_grdiv, axis=E2C2EODim)
    return astype(z_graddiv2_vn_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_graddiv2_of_vn(
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], wpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
    squared_horizontal_gradient_of_total_divergence: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_graddiv2_of_vn(
        geofac_grdiv,
        horizontal_gradient_of_normal_wind_divergence,
        out=squared_horizontal_gradient_of_total_divergence,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
