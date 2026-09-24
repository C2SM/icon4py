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
from icon4py.model.common.math.tridiagonal import (
    _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKHalfField[wpfloat],
    ddqz_z_half: fa.CellKHalfField[vpfloat],
    z_alpha: fa.CellKHalfField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKHalfField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_52."""
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)

    z_gamma_vp = astype(dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half_wp, vpfloat)
    z_a = (vpfloat("0.0") - z_gamma_vp) * z_beta(dims.KHalfDim - 0.5) * z_alpha(dims.KHalfDim - 1)
    z_c = (vpfloat("0.0") - z_gamma_vp) * z_beta(dims.KHalfDim + 0.5) * z_alpha(dims.KHalfDim + 1)
    z_b = vpfloat("1.0") + z_gamma_vp * z_alpha * (
        z_beta(dims.KHalfDim - 0.5) + z_beta(dims.KHalfDim + 0.5)
    )
    z_gamma_wp = astype(z_gamma_vp, wpfloat)
    w_prep = z_w_expl - z_gamma_wp * (
        z_exner_expl(dims.KHalfDim - 0.5) - z_exner_expl(dims.KHalfDim + 0.5)
    )
    z_q_res, w_res = _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision(
        a=z_a, b=z_b, c=z_c, d=w_prep
    )
    return z_q_res, w_res


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKHalfField[wpfloat],
    ddqz_z_half: fa.CellKHalfField[vpfloat],
    z_alpha: fa.CellKHalfField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKHalfField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    z_q: fa.CellKHalfField[vpfloat],
    w: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_tridiagonal_matrix_for_w_forward_sweep(
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_ic=theta_v_ic,
        ddqz_z_half=ddqz_z_half,
        z_alpha=z_alpha,
        z_beta=z_beta,
        z_w_expl=z_w_expl,
        z_exner_expl=z_exner_expl,
        dtime=dtime,
        cpd=cpd,
        out=(z_q, w),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
