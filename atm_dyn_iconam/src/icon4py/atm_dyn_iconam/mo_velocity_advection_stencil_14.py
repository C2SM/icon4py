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
from functional.ffront.fbuiltins import Field, where

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_velocity_advection_stencil_14_z_w_con_c(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_clipping: Field[[CellDim, KDim], float],
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    vcfl = z_w_con_c * dtime / ddqz_z_half
    z_w_con_c = where(
        (cfl_clipping == 1.0) & (vcfl < -0.85), -0.85 * ddqz_z_half / dtime, z_w_con_c
    )
    z_w_con_c = where(
        (cfl_clipping == 1.0) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c
    )
    return z_w_con_c


@field_operator
def _mo_velocity_advection_stencil_14_cfl_clipping(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
) -> Field[[CellDim, KDim], float]:
    cfl_clipping = where(z_w_con_c > cfl_w_limit * ddqz_z_half, 1.0, 0.0)
    return cfl_clipping


@field_operator
def _mo_velocity_advection_stencil_14_pre_levelmask(
    cfl_clipping: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], bool]:
    pre_levelmask = where(cfl_clipping == 1.0, 1.0, 0.0)
    return pre_levelmask


@field_operator
def _mo_velocity_advection_stencil_14_vcfl(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_clipping: Field[[CellDim, KDim], float],
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    vcfl = where(cfl_clipping == 1.0, z_w_con_c * dtime / ddqz_z_half, 0.0)
    return vcfl


@program
def mo_velocity_advection_stencil_14(
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    cfl_clipping: Field[[CellDim, KDim], float],
    pre_levelmask: Field[[CellDim, KDim], bool],
    vcfl: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
):
    _mo_velocity_advection_stencil_14_cfl_clipping(
        ddqz_z_half, z_w_con_c, cfl_w_limit, out=cfl_clipping
    )

    _mo_velocity_advection_stencil_14_pre_levelmask(cfl_clipping, out=pre_levelmask)

    _mo_velocity_advection_stencil_14_vcfl(
        ddqz_z_half, z_w_con_c, cfl_clipping, dtime, out=vcfl
    )

    _mo_velocity_advection_stencil_14_z_w_con_c(
        ddqz_z_half, z_w_con_c, cfl_clipping, dtime, out=z_w_con_c
    )
