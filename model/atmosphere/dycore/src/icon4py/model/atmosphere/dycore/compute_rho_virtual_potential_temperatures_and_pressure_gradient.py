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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
    ddqz_z_half: fa.CKvpField,
    rho_now: fa.CKwpField,
    rho_var: fa.CKwpField,
    theta_now: fa.CKwpField,
    theta_var: fa.CKwpField,
    wgtfac_c: fa.CKvpField,
    theta_ref_mc: fa.CKvpField,
    vwind_expl_wgt: fa.CwpField,
    exner_pr: fa.CKwpField,
    d_exner_dz_ref_ic: fa.CKvpField,
    dtime: wpfloat,
    wgt_nnow_rth: wpfloat,
    wgt_nnew_rth: wpfloat,
) -> tuple[
    fa.CKwpField,
    fa.CKvpField,
    fa.CKwpField,
    fa.CKvpField,
]:
    """Formerly known as _mo_solve_nonhydro_stencil_10."""
    w_concorr_c_wp, wgtfac_c_wp, theta_ref_mc_wp, ddqz_z_half_wp = astype(
        (w_concorr_c, wgtfac_c, theta_ref_mc, ddqz_z_half), wpfloat
    )

    z_w_backtraj_wp = -(w - w_concorr_c_wp) * dtime * wpfloat("0.5") / ddqz_z_half_wp
    z_rho_tavg_m1_wp = wgt_nnow_rth * rho_now(Koff[-1]) + wgt_nnew_rth * rho_var(Koff[-1])
    z_theta_tavg_m1_wp = wgt_nnow_rth * theta_now(Koff[-1]) + wgt_nnew_rth * theta_var(Koff[-1])
    z_rho_tavg_wp = wgt_nnow_rth * rho_now + wgt_nnew_rth * rho_var
    z_theta_tavg_wp = wgt_nnow_rth * theta_now + wgt_nnew_rth * theta_var
    rho_ic_wp = (
        wgtfac_c_wp * z_rho_tavg_wp
        + (wpfloat("1.0") - wgtfac_c_wp) * z_rho_tavg_m1_wp
        + z_w_backtraj_wp * (z_rho_tavg_m1_wp - z_rho_tavg_wp)
    )
    z_theta_v_pr_mc_m1_wp = z_theta_tavg_m1_wp - theta_ref_mc_wp(Koff[-1])
    z_theta_v_pr_mc_wp = z_theta_tavg_wp - theta_ref_mc_wp

    z_theta_v_pr_mc_vp, z_theta_v_pr_mc_m1_vp = astype(
        (z_theta_v_pr_mc_wp, z_theta_v_pr_mc_m1_wp), vpfloat
    )
    z_theta_v_pr_ic_vp = (
        wgtfac_c * z_theta_v_pr_mc_vp + (vpfloat("1.0") - wgtfac_c) * z_theta_v_pr_mc_m1_vp
    )

    theta_v_ic_wp = (
        wgtfac_c_wp * z_theta_tavg_wp
        + (wpfloat("1.0") - wgtfac_c_wp) * z_theta_tavg_m1_wp
        + z_w_backtraj_wp * (z_theta_tavg_m1_wp - z_theta_tavg_wp)
    )
    z_th_ddz_exner_c_wp = vwind_expl_wgt * theta_v_ic_wp * (exner_pr(Koff[-1]) - exner_pr) / astype(
        ddqz_z_half, wpfloat
    ) + astype(z_theta_v_pr_ic_vp * d_exner_dz_ref_ic, wpfloat)
    return (
        rho_ic_wp,
        z_theta_v_pr_ic_vp,
        theta_v_ic_wp,
        astype(z_th_ddz_exner_c_wp, vpfloat),
    )


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
    ddqz_z_half: fa.CKvpField,
    rho_now: fa.CKwpField,
    rho_var: fa.CKwpField,
    theta_now: fa.CKwpField,
    theta_var: fa.CKwpField,
    wgtfac_c: fa.CKvpField,
    theta_ref_mc: fa.CKvpField,
    vwind_expl_wgt: fa.CwpField,
    exner_pr: fa.CKwpField,
    d_exner_dz_ref_ic: fa.CKvpField,
    rho_ic: fa.CKwpField,
    z_theta_v_pr_ic: fa.CKvpField,
    theta_v_ic: fa.CKwpField,
    z_th_ddz_exner_c: fa.CKvpField,
    dtime: wpfloat,
    wgt_nnow_rth: wpfloat,
    wgt_nnew_rth: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
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
        out=(
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
