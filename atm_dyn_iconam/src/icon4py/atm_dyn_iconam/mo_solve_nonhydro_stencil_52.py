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

from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import Field, where

from icon4py.common.dimension import CellDim, KDim, Koff


"""
TODO
The implementation of this computation relies on implementation details of GT4Py.

All accesses to `Koff[-1]` in `_mo_solve_nonhydro_stencil_52` are out of bounds,
as the domain starts at `k==0` in this implementation. They would need to be protected for `k==0`.

However, the values at `k==0` are never accessed in the scan. If we force-inlining
of these accesses into the `if_` in the scan, we will not do out-of-bounds accesses.
Current implementation in GT4Py with the default `FORCE_INLINE` option will do that.

Alternatives that are currently not implementable:
1) Apply the offset `Koff[-1]` inside the scan in the protected branch.
   This is not implementable as the `scan_operator` in field view is a scalar operator.
   We could change that in the future.
2) Protect the `Koff[-1]` accesses for `k==0`, e.g. `where(k==0, NaN, the_expression_with_shift)`.
   We could implement that, however with potential performance loss as we don't have optimizations
   that merge that condition with the condition in the scan.
   Additionally, as we don't have an `index()`-builtin, we would have to pass explicitly
   a field providing the index.
"""


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, True))
def _w(
    state: tuple[float, float, bool],
    w: float,
    z_q: float,
    z_a: float,
    z_b: float,
    z_c: float,
    w_prep: float,
):
    z_q_m1, w_m1, first = state
    z_g = 1.0 / (z_b + z_a * z_q_m1)
    z_q_new = (0.0 - z_c) * z_g
    w_new = (w_prep - z_a * w_m1) * z_g
    return (z_q, w, False) if first else (z_q_new, w_new, False)


@field_operator
def _mo_solve_nonhydro_stencil_52_relying_on_inlining(
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    z_w_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    z_q: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    dtime: float,
    cpd: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = (0.0 - z_gamma) * z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = (0.0 - z_gamma) * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    w_prep = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    z_q_res, w_res, _ = _w(w, z_q, z_a, z_b, z_c, w_prep)
    return z_q_res, w_res


@field_operator
def _mo_solve_nonhydro_stencil_52(
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    z_w_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    k_index: Field[[KDim], int],
    z_q: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    dtime: float,
    cpd: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    first_level = k_index == 0
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = (
        (0.0 - z_gamma)
        * where(first_level, 0.0, z_beta(Koff[-1]))
        * where(first_level, 0.0, z_alpha(Koff[-1]))
    )
    z_c = (0.0 - z_gamma) * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (where(first_level, 0.0, z_beta(Koff[-1])) + z_beta)
    w_prep = z_w_expl - z_gamma * (
        where(first_level, 0.0, z_exner_expl(Koff[-1])) - z_exner_expl
    )
    z_q_res, w_res, _ = _w(w, z_q, z_a, z_b, z_c, w_prep)
    return z_q_res, w_res


@program
def mo_solve_nonhydro_stencil_52(
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    z_w_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    z_q: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    dtime: float,
    cpd: float,
):
    _mo_solve_nonhydro_stencil_52(
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
    )
