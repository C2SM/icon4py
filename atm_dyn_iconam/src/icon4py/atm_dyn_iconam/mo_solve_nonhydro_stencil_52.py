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

from functional.ffront.decorator import field_operator, program, scan_operator
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim, Koff


@scan_operator(axis=KDim, forward=False, init=0.0)
def _mo_solve_nonhydro_stencil_52_scan(
    w_state: float, z_a: float, z_g: float, w: float
) -> float:
    return (w - z_a * w_state) * z_g


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
    dtime: float,
    cpd: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = -z_gamma * z_beta(Koff[-1]) * z_alpha(Koff[1])
    z_c = -z_gamma * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_g = 1.0 / (z_b + z_a * z_q(Koff[-1]))
    z_q = -z_c * z_g
    w = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    w = _mo_solve_nonhydro_stencil_52_scan(z_a, z_g, w)
    return z_q, w


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
        dtime,
        cpd,
        out=(z_q, w),
    )
