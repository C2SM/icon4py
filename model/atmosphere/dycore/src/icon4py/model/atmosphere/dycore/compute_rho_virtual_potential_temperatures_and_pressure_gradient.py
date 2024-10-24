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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    rho_now: fa.CellKField[wpfloat],
    rho_var: fa.CellKField[wpfloat],
    theta_now: fa.CellKField[wpfloat],
    theta_var: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    dtime: wpfloat,
    wgt_nnow_rth: wpfloat,
    wgt_nnew_rth: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
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


@program(grid_type=GridType.UNSTRUCTURED)
def compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    rho_now: fa.CellKField[wpfloat],
    rho_var: fa.CellKField[wpfloat],
    theta_now: fa.CellKField[wpfloat],
    theta_var: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    dtime: wpfloat,
    wgt_nnow_rth: wpfloat,
    wgt_nnew_rth: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
