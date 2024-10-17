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
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_math_gradients_grad_green_gauss_cell_dsl(
    p_ccpr1: fa.CellKField[vpfloat],
    p_ccpr2: fa.CellKField[vpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    p_ccpr1_wp, p_ccpr2_wp = astype((p_ccpr1, p_ccpr2), wpfloat)

    p_grad_1_u_wp = neighbor_sum(geofac_grg_x * p_ccpr1_wp(C2E2CO), axis=C2E2CODim)
    p_grad_1_v_wp = neighbor_sum(geofac_grg_y * p_ccpr1_wp(C2E2CO), axis=C2E2CODim)
    p_grad_2_u_wp = neighbor_sum(geofac_grg_x * p_ccpr2_wp(C2E2CO), axis=C2E2CODim)
    p_grad_2_v_wp = neighbor_sum(geofac_grg_y * p_ccpr2_wp(C2E2CO), axis=C2E2CODim)
    return astype((p_grad_1_u_wp, p_grad_1_v_wp, p_grad_2_u_wp, p_grad_2_v_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_math_gradients_grad_green_gauss_cell_dsl(
    p_grad_1_u: fa.CellKField[vpfloat],
    p_grad_1_v: fa.CellKField[vpfloat],
    p_grad_2_u: fa.CellKField[vpfloat],
    p_grad_2_v: fa.CellKField[vpfloat],
    p_ccpr1: fa.CellKField[vpfloat],
    p_ccpr2: fa.CellKField[vpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _mo_math_gradients_grad_green_gauss_cell_dsl(
        p_ccpr1,
        p_ccpr2,
        geofac_grg_x,
        geofac_grg_y,
        out=(p_grad_1_u, p_grad_1_v, p_grad_2_u, p_grad_2_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
