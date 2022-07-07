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


@field_operator
def _mo_solve_nonhydro_stencil_61_rho_new(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    rho_new = rho_now + dtime * grf_tend_rho
    return rho_new


@field_operator
def _mo_solve_nonhydro_stencil_61_exner_new(
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    exner_new = theta_v_now + dtime * grf_tend_thv
    return exner_new


@field_operator
def _mo_solve_nonhydro_stencil_61_w_new(
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    dtime: float,
) -> Field[[CellDim, KDim], float]:
    w_new = w_now + dtime * grf_tend_w
    return w_new


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
    dtime: float,
):
    _mo_solve_nonhydro_stencil_61_rho_new(rho_now, grf_tend_rho, dtime, out=rho_new)
    _mo_solve_nonhydro_stencil_61_exner_new(
        theta_v_now, grf_tend_thv, dtime, out=exner_new
    )
    _mo_solve_nonhydro_stencil_61_w_new(w_now, grf_tend_w, dtime, out=w_new)
