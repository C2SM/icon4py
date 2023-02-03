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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.common.dimension import CellDim, KDim, Koff


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, int32(1)))
def _w(
    state: tuple[float, float, int32],
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
    return (z_q, w, int32(0)) if first == int32(1) else (z_q_new, w_new, int32(0))


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
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], int32],
]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = (0.0 - z_gamma) * z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = (0.0 - z_gamma) * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    w_prep = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    z_q_res, w_res, dummy_bool = _w(w, z_q, z_a, z_b, z_c, w_prep)
    return z_q_res, w_res, dummy_bool


@field_operator
def _mo_solve_nonhydro_stencil_52_z_q(
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
) -> Field[[CellDim, KDim], float]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = -z_gamma * z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = -z_gamma * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    w_prep = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    z_q_res, w_res, _ = _w(w, z_q, z_a, z_b, z_c, w_prep)
    return z_q_res


@field_operator
def _mo_solve_nonhydro_stencil_52_w(
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
) -> Field[[CellDim, KDim], float]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = -z_gamma * z_beta(Koff[-1]) * z_alpha(Koff[-1])
    z_c = -z_gamma * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    w_prep = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    z_q_res, w_res, _ = _w(w, z_q, z_a, z_b, z_c, w_prep)
    return w_res


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
    dummy_bool: Field[[CellDim, KDim], int32],
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
        out=(z_q[:, 1:], w[:, 1:], dummy_bool[:, 1:]),
    )
