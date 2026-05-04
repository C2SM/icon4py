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
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_results_for_thermodynamic_variables(
    rho_explicit_term: fa.CellKField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    exner_explicit_term: fa.CellKField[wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[vpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    current_exner: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_55."""
    inv_ddqz_z_full_wp, reference_exner_at_cells_on_model_levels_wp, z_alpha_wp, z_beta_wp = astype(
        (inv_ddqz_z_full, reference_exner_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels, tridiagonal_beta_coeff_at_cells_on_model_levels), wpfloat
    )

    rho_new_wp = rho_explicit_term - exner_w_implicit_weight_parameter * dtime * inv_ddqz_z_full_wp * (
        rho_at_cells_on_half_levels * w - rho_at_cells_on_half_levels(Koff[1]) * w(Koff[1])
    )
    exner_new_wp = (
        exner_explicit_term
        + reference_exner_at_cells_on_model_levels_wp
        - z_beta_wp * (z_alpha_wp * w - z_alpha_wp(Koff[1]) * w(Koff[1]))
    )
    theta_v_new_wp = (
        current_rho
        * current_theta_v
        * ((exner_new_wp / current_exner - wpfloat("1.0")) * PhysicsConstants.cvd_o_rd + wpfloat("1.0"))
        / rho_new_wp
    )
    return rho_new_wp, exner_new_wp, theta_v_new_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_results_for_thermodynamic_variables(
    rho_explicit_term: fa.CellKField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    exner_explicit_term: fa.CellKField[wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[vpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    current_exner: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    theta_v_new: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_results_for_thermodynamic_variables(
        rho_explicit_term,
        exner_w_implicit_weight_parameter,
        inv_ddqz_z_full,
        rho_at_cells_on_half_levels,
        w,
        exner_explicit_term,
        reference_exner_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        current_rho,
        current_theta_v,
        current_exner,
        dtime,
        out=(rho_new, exner_new, theta_v_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
