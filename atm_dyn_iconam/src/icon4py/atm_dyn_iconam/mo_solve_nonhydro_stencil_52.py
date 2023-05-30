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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim, Koff


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, True))
def _w(
    state: tuple[float, float, bool],
    w_prev: float,  # only accessed at the first k-level
    z_q_prev: float,
    z_a: float,
    z_b: float,
    z_c: float,
    w_prep: float,
):
    first = state[2]
    z_q_m1 = z_q_prev if first else state[0]
    w_m1 = w_prev if first else state[1]
    z_g = 1.0 / (z_b + z_a * z_q_m1)
    z_q_new = (0.0 - z_c) * z_g
    w_new = (w_prep - z_a * w_m1) * z_g
    return z_q_new, w_new, False


@field_operator
def _mo_solve_nonhydro_stencil_52(
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
    w_prev = w(Koff[-1])
    z_q_prev = z_q(Koff[-1])
    z_q_res, w_res, _ = _w(w_prev, z_q_prev, z_a, z_b, z_c, w_prep)
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
        out=(z_q[:, 1:], w[:, 1:]),
    )
