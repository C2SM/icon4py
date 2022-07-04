# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, float32, neighbor_sum

from icon4py.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u(
    p_ccpr1: Field[[CellDim, KDim], float32],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float32],
) -> Field[[CellDim, KDim], float32]:
    p_grad_1_u = neighbor_sum(p_ccpr1(C2E2CO) * geofac_grg_x, axis=C2E2CODim)
    return p_grad_1_u


@program
def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u(
    p_ccpr1: Field[[CellDim, KDim], float32],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float32],
    p_grad_1_u: Field[[CellDim, KDim], float32],
):
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u(
        p_ccpr1, geofac_grg_x, out=p_grad_1_u
    )


@field_operator
def _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v(
    p_ccpr1: Field[[CellDim, KDim], float32],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float32],
) -> Field[[CellDim, KDim], float32]:
    p_grad_1_v = neighbor_sum(p_ccpr1(C2E2CO) * geofac_grg_y, axis=C2E2CODim)
    return p_grad_1_v


@program
def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v(
    p_ccpr1: Field[[CellDim, KDim], float32],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float32],
    p_grad_1_v: Field[[CellDim, KDim], float32],
):
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v(
        p_ccpr1, geofac_grg_y, out=p_grad_1_v
    )


@field_operator
def _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u(
    p_ccpr2: Field[[CellDim, KDim], float32],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float32],
) -> Field[[CellDim, KDim], float32]:
    p_grad_2_u = neighbor_sum(p_ccpr2(C2E2CO) * geofac_grg_x, axis=C2E2CODim)
    return p_grad_2_u


@program
def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u(
    p_ccpr2: Field[[CellDim, KDim], float32],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float32],
    p_grad_2_u: Field[[CellDim, KDim], float32],
):
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u(
        p_ccpr2, geofac_grg_x, out=p_grad_2_u
    )


@field_operator
def _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v(
    p_ccpr2: Field[[CellDim, KDim], float32],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float32],
) -> Field[[CellDim, KDim], float32]:
    p_grad_2_v = neighbor_sum(p_ccpr2(C2E2CO) * geofac_grg_y, axis=C2E2CODim)
    return p_grad_2_v


@program
def mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v(
    p_ccpr2: Field[[CellDim, KDim], float32],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float32],
    p_grad_2_v: Field[[CellDim, KDim], float32],
):
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v(
        p_ccpr2, geofac_grg_y, out=p_grad_2_v
    )


@program
def mo_math_gradients_grad_green_gauss_cell_dsl(
    p_grad_1_u: Field[[CellDim, KDim], float32],
    p_grad_1_v: Field[[CellDim, KDim], float32],
    p_grad_2_u: Field[[CellDim, KDim], float32],
    p_grad_2_v: Field[[CellDim, KDim], float32],
    p_ccpr1: Field[[CellDim, KDim], float32],
    p_ccpr2: Field[[CellDim, KDim], float32],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float32],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float32],
):
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_u(
        p_ccpr1, geofac_grg_x, out=p_grad_1_u
    )
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_1_v(
        p_ccpr1, geofac_grg_y, out=p_grad_1_v
    )
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_u(
        p_ccpr2, geofac_grg_x, out=p_grad_2_u
    )
    _mo_math_gradients_grad_green_gauss_cell_dsl_p_grad_2_v(
        p_ccpr2, geofac_grg_y, out=p_grad_2_v
    )
