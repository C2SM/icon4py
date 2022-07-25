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
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


# TODO: change dtime back to type float


@field_operator
def _mo_solve_nonhydro_stencil_61(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    dtime: Field[[CellDim, KDim], float],
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    rho_new = rho_now + dtime * grf_tend_rho
    exner_new = theta_v_now + dtime * grf_tend_thv
    w_new = w_now + dtime * grf_tend_w
    return rho_new, exner_new, w_new


@field_operator
def _mo_solve_nonhydro_stencil_61_rho_new(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    dtime: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_61(
        rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w, dtime
    )[0]


@field_operator
def _mo_solve_nonhydro_stencil_61_exner_new(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    dtime: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_61(
        rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w, dtime
    )[1]


@field_operator
def _mo_solve_nonhydro_stencil_61_w_new(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    dtime: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_61(
        rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w, dtime
    )[2]


@program
def mo_solve_nonhydro_stencil_61(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    dtime: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_61_rho_new(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=rho_new,
    )
    _mo_solve_nonhydro_stencil_61_exner_new(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=exner_new,
    )
    _mo_solve_nonhydro_stencil_61_w_new(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
    )
