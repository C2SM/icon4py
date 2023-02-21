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

from icon4py.common.dimension import CellDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_10(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    rho_var: Field[[CellDim, KDim], float],
    theta_now: Field[[CellDim, KDim], float],
    theta_var: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    z_w_backtraj = -(w - w_concorr_c) * dtime * 0.5 / ddqz_z_half
    z_rho_tavg_m1 = wgt_nnow_rth * rho_now(Koff[-1]) + wgt_nnew_rth * rho_var(Koff[-1])
    z_theta_tavg_m1 = wgt_nnow_rth * theta_now(Koff[-1]) + wgt_nnew_rth * theta_var(
        Koff[-1]
    )
    z_rho_tavg = wgt_nnow_rth * rho_now + wgt_nnew_rth * rho_var
    z_theta_tavg = wgt_nnow_rth * theta_now + wgt_nnew_rth * theta_var
    rho_ic = (
        wgtfac_c * z_rho_tavg
        + (1.0 - wgtfac_c) * z_rho_tavg_m1
        + z_w_backtraj * (z_rho_tavg_m1 - z_rho_tavg)
    )
    z_theta_v_pr_mc_m1 = z_theta_tavg_m1 - theta_ref_mc(Koff[-1])
    z_theta_v_pr_mc = z_theta_tavg - theta_ref_mc
    z_theta_v_pr_ic = wgtfac_c * z_theta_v_pr_mc + (1.0 - wgtfac_c) * z_theta_v_pr_mc_m1
    theta_v_ic = (
        wgtfac_c * z_theta_tavg
        + (1.0 - wgtfac_c) * z_theta_tavg_m1
        + z_w_backtraj * (z_theta_tavg_m1 - z_theta_tavg)
    )
    z_th_ddz_exner_c = (
        vwind_expl_wgt * theta_v_ic * (exner_pr(Koff[-1]) - exner_pr) / ddqz_z_half
        + z_theta_v_pr_ic * d_exner_dz_ref_ic
    )
    return (
        rho_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
    )


@program
def mo_solve_nonhydro_stencil_10(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    rho_var: Field[[CellDim, KDim], float],
    theta_now: Field[[CellDim, KDim], float],
    theta_var: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
):
    _mo_solve_nonhydro_stencil_10(
        w,
        w_concorr_c,
        ddqz_z_half,
        rho_now,
        rho_var,
        theta_now,
        theta_var,
        wgtfac_c,
        theta_ref_mc,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        out=(rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )
