# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    current_rho: fa.CellKField[wpfloat],
    rho_var: fa.CellKField[wpfloat],
    theta_now: fa.CellKField[wpfloat],
    theta_var: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[wpfloat],
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
    w_concorr_c_wp, wgtfac_c_wp, reference_theta_at_cells_on_model_levels_wp, ddqz_z_half_wp = astype(
        (contravariant_correction_at_cells_on_half_levels, wgtfac_c, reference_theta_at_cells_on_model_levels, ddqz_z_half), wpfloat
    )

    z_w_backtraj_wp = -(w - w_concorr_c_wp) * dtime * wpfloat("0.5") / ddqz_z_half_wp
    z_rho_tavg_m1_wp = wgt_nnow_rth * current_rho(Koff[-1]) + wgt_nnew_rth * rho_var(Koff[-1])
    z_theta_tavg_m1_wp = wgt_nnow_rth * theta_now(Koff[-1]) + wgt_nnew_rth * theta_var(Koff[-1])
    z_rho_tavg_wp = wgt_nnow_rth * current_rho + wgt_nnew_rth * rho_var
    z_theta_tavg_wp = wgt_nnow_rth * theta_now + wgt_nnew_rth * theta_var
    rho_ic_wp = (
        wgtfac_c_wp * z_rho_tavg_wp
        + (wpfloat("1.0") - wgtfac_c_wp) * z_rho_tavg_m1_wp
        + z_w_backtraj_wp * (z_rho_tavg_m1_wp - z_rho_tavg_wp)
    )
    z_theta_v_pr_mc_m1_wp = z_theta_tavg_m1_wp - reference_theta_at_cells_on_model_levels_wp(Koff[-1])
    z_theta_v_pr_mc_wp = z_theta_tavg_wp - reference_theta_at_cells_on_model_levels_wp

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
    z_th_ddz_exner_c_wp = exner_w_explicit_weight_parameter * theta_v_ic_wp * (perturbed_exner_at_cells_on_model_levels(Koff[-1]) - perturbed_exner_at_cells_on_model_levels) / astype(
        ddqz_z_half, wpfloat
    ) + astype(z_theta_v_pr_ic_vp * d_exner_dz_ref_ic, wpfloat)
    return (
        rho_ic_wp,
        z_theta_v_pr_ic_vp,
        theta_v_ic_wp,
        astype(z_th_ddz_exner_c_wp, vpfloat),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_rho_virtual_potential_temperatures_and_pressure_gradient(
    w: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    current_rho: fa.CellKField[wpfloat],
    rho_var: fa.CellKField[wpfloat],
    theta_now: fa.CellKField[wpfloat],
    theta_var: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    dtime: wpfloat,
    wgt_nnow_rth: wpfloat,
    wgt_nnew_rth: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_rho_virtual_potential_temperatures_and_pressure_gradient(
        w,
        contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half,
        current_rho,
        rho_var,
        theta_now,
        theta_var,
        wgtfac_c,
        reference_theta_at_cells_on_model_levels,
        exner_w_explicit_weight_parameter,
        perturbed_exner_at_cells_on_model_levels,
        d_exner_dz_ref_ic,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        out=(
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
