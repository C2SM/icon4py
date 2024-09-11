# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@scan_operator(axis=dims.KDim, forward=True, init=(vpfloat("0.0"), 0.0, True))
def _w(
    state: tuple[vpfloat, float, bool],
    w_prev: wpfloat,  # only accessed at the first k-level
    z_q_prev: vpfloat,
    z_a: vpfloat,
    z_b: vpfloat,
    z_c: vpfloat,
    w_prep: wpfloat,
):
    first = state[2]
    z_q_m1 = z_q_prev if first else astype(state[0], vpfloat)
    w_m1 = w_prev if first else state[1]
    z_g = vpfloat("1.0") / (z_b + z_a * z_q_m1)
    z_q_new = (vpfloat("0.0") - z_c) * z_g
    w_new = (w_prep - astype(z_a, wpfloat) * w_m1) * astype(z_g, wpfloat)
    return z_q_new, w_new, False


@field_operator
def _solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    z_q: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_52."""
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)

    z_gamma_vp = astype(dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half_wp, vpfloat)
    z_a = (vpfloat("0.0") - z_gamma_vp) * z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = (vpfloat("0.0") - z_gamma_vp) * z_beta * z_alpha(Koff[1])
    z_b = vpfloat("1.0") + z_gamma_vp * z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_gamma_wp = astype(z_gamma_vp, wpfloat)
    w_prep = z_w_expl - z_gamma_wp * (z_exner_expl(Koff[-1]) - z_exner_expl)
    w_prev = w(Koff[-1])
    z_q_prev = z_q(Koff[-1])
    z_q_res, w_res, _ = _w(w_prev, z_q_prev, z_a, z_b, z_c, w_prep)
    return z_q_res, w_res


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    z_q: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _solve_tridiagonal_matrix_for_w_forward_sweep(
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_alpha,
        z_beta,
        z_w_expl,
        z_exner_expl,
        z_q,
        w,
        dtime,
        cpd,
        out=(z_q, w),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
