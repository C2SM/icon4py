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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_solver_coefficients_matrix(
    current_exner: fa.CellKField[wpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_44."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_beta_wp = dtime * rd * current_exner / (cvd * current_rho * current_theta_v) * inv_ddqz_z_full_wp
    z_alpha_wp = exner_w_implicit_weight_parameter * theta_v_at_cells_on_half_levels * rho_at_cells_on_half_levels
    return astype((z_beta_wp, z_alpha_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_solver_coefficients_matrix(
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    current_exner: fa.CellKField[wpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_solver_coefficients_matrix(
        current_exner,
        current_rho,
        current_theta_v,
        inv_ddqz_z_full,
        exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels,
        rho_at_cells_on_half_levels,
        dtime,
        rd,
        cvd,
        out=(tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
