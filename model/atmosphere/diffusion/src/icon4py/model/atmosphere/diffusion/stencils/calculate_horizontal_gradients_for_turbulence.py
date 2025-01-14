# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    dwdx_wp = neighbor_sum(geofac_grg_x * w(C2E2CO), axis=C2E2CODim)
    dwdy_wp = neighbor_sum(geofac_grg_y * w(C2E2CO), axis=C2E2CODim)
    return astype((dwdx_wp, dwdy_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_horizontal_gradients_for_turbulence(
    w: fa.CellKField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_horizontal_gradients_for_turbulence(
        w,
        geofac_grg_x,
        geofac_grg_y,
        out=(dwdx, dwdy),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
