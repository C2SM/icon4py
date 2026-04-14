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


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=(
        vpfloat("0.0"),
        0.0,
    ),  # boundary condition for upper tridiagonal element and w at model top
)
def tridiagonal_forward_sweep_for_w(
    state_kminus1: tuple[vpfloat, float],
    a: vpfloat,
    b: vpfloat,
    c: vpfloat,
    d: wpfloat,
):
    """
    |  1   0                  |  | w_0 |    |  0  |          | 1   0                     |  | w_0 |    |  0     |
    | a_1 b_1 c_1             |  | w_1 |    | d_1 |          | 0   1  cnew_1             |  | w_1 |    | dnew_1 |
    |     a_2 b_2 c_2         |  | w_2 |  = | d_2 |    ==>   |     0   1  cnew_2         |  | w_2 |  = | dnew_2 |
    |         a_3 b_3 c_3     |  | w_3 |    | d_3 |          |         0   1  cnew_3     |  | w_3 |    | dnew_3 |
    |             a_4 b_4 c_4 |  | w_4 |    | d_4 |          |             0   1  cnew_4 |  | w_4 |    | dnew_4 |
    |                 ...     |  | ... |    | ... |          |                 ...       |  | ... |    | ...    |
    """
    c_kminus1 = astype(state_kminus1[0], vpfloat)
    d_kminus1 = state_kminus1[1]
    normalization = vpfloat("1.0") / (b + a * c_kminus1)  # normalize diagonal element to 1
    c_new = (vpfloat("0.0") - c) * normalization
    d_new = (d - astype(a, wpfloat) * d_kminus1) * astype(normalization, wpfloat)
    return c_new, d_new


@gtx.field_operator
def _solve_tridiagonal_matrix_for_w_forward_sweep(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[vpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    w_explicit_term: fa.CellKField[wpfloat],
    exner_explicit_term: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_52."""
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)

    z_gamma_vp = astype(dtime * cpd * exner_w_implicit_weight_parameter * theta_v_at_cells_on_half_levels / ddqz_z_half_wp, vpfloat)
    z_a = (vpfloat("0.0") - z_gamma_vp) * tridiagonal_beta_coeff_at_cells_on_model_levels(Koff[-1]) * tridiagonal_alpha_coeff_at_cells_on_half_levels(Koff[-1])
    z_c = (vpfloat("0.0") - z_gamma_vp) * tridiagonal_beta_coeff_at_cells_on_model_levels * tridiagonal_alpha_coeff_at_cells_on_half_levels(Koff[1])
    z_b = vpfloat("1.0") + z_gamma_vp * tridiagonal_alpha_coeff_at_cells_on_half_levels * (tridiagonal_beta_coeff_at_cells_on_model_levels(Koff[-1]) + tridiagonal_beta_coeff_at_cells_on_model_levels)
    z_gamma_wp = astype(z_gamma_vp, wpfloat)
    w_prep = w_explicit_term - z_gamma_wp * (exner_explicit_term(Koff[-1]) - exner_explicit_term)
    z_q_res, w_res = tridiagonal_forward_sweep_for_w(z_a, z_b, z_c, w_prep)
    return z_q_res, w_res


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_tridiagonal_matrix_for_w_forward_sweep(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[vpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    w_explicit_term: fa.CellKField[wpfloat],
    exner_explicit_term: fa.CellKField[wpfloat],
    tridiagonal_intermediate_result: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_tridiagonal_matrix_for_w_forward_sweep(
        exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels,
        ddqz_z_half,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        w_explicit_term,
        exner_explicit_term,
        dtime,
        cpd,
        out=(tridiagonal_intermediate_result, w),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
