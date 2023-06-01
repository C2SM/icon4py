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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field
from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_55(
    z_rho_expl: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    dtime: float,
    cvd_o_rd: float,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    rho_new = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full * (
        rho_ic * w - rho_ic(Koff[1]) * w(Koff[1])
    )
    exner_new = (
        z_exner_expl
        + exner_ref_mc
        - z_beta * (z_alpha * w - z_alpha(Koff[1]) * w(Koff[1]))
    )
    theta_v_new = (
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - 1.0) * cvd_o_rd + 1.0)
        / rho_new
    )
    return rho_new, exner_new, theta_v_new


@program
def mo_solve_nonhydro_stencil_55(
    z_rho_expl: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    dtime: float,
    cvd_o_rd: float,
):
    _mo_solve_nonhydro_stencil_55(
        z_rho_expl,
        vwind_impl_wgt,
        inv_ddqz_z_full,
        rho_ic,
        w,
        z_exner_expl,
        exner_ref_mc,
        z_alpha,
        z_beta,
        rho_now,
        theta_v_now,
        exner_now,
        dtime,
        cvd_o_rd,
        out=(rho_new, exner_new, theta_v_new),
    )
