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
from functional.ffront.fbuiltins import Field, float

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_44_z_beta(
    exner_nnow: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    dtime: float,
    rd: float,
    cvd: float,
) -> Field[[CellDim, KDim], float]:
    z_beta = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full
    return z_beta


@program
def mo_solve_nonhydro_stencil_44_z_beta(
    z_beta: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    dtime: float,
    rd: float,
    cvd: float,
):
    _mo_solve_nonhydro_stencil_44_z_beta(
        exner_nnow, rho_nnow, theta_v_nnow, inv_ddqz_z_full, dtime, rd, cvd, out=z_beta
    )


@field_operator
def _mo_solve_nonhydro_stencil_44_z_alpha(
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_alpha = vwind_impl_wgt * theta_v_ic * rho_ic
    return z_alpha


@program
def mo_solve_nonhydro_stencil_44_z_alpha(
    z_alpha: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_44_z_alpha(
        vwind_impl_wgt, theta_v_ic, rho_ic, out=z_alpha
    )


def mo_solve_nonhydro_stencil_44(
    z_beta: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    dtime: float,
    rd: float,
    cvd: float,
):
    _mo_solve_nonhydro_stencil_44_z_beta(
        exner_nnow, rho_nnow, theta_v_nnow, inv_ddqz_z_full, dtime, rd, cvd, out=z_beta
    )
    _mo_solve_nonhydro_stencil_44_z_alpha(
        vwind_impl_wgt, theta_v_ic, rho_ic, out=z_alpha
    )
