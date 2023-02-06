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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, as_offset

from icon4py.common.dimension import (
    E2C,
    E2EC,
    CellDim,
    ECDim,
    EdgeDim,
    KDim,
    Koff,
)


@field_operator
def _mo_solve_nonhydro_stencil_21(
    theta_v: Field[[CellDim, KDim], float],
    ikidx: Field[[ECDim, KDim], float],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    grav_o_cpd: float,
) -> Field[[EdgeDim, KDim], float]:
    theta_v_0, theta_v_1 = theta_v(E2C[0]), theta_v(E2C[1])
    theta_v_ic_0, theta_v_ic_1 = theta_v_ic(E2C[0]), theta_v_ic(E2C[1])
    inv_ddqz_z_full_0, inv_ddqz_z_full_1 = inv_ddqz_z_full(E2C[0]), inv_ddqz_z_full(
        E2C[1]
    )
    z_theta1 = (
        theta_v_0(as_offset(Koff, ikidx(E2EC[0])))
        + zdiff_gradp(E2EC[0]) * theta_v_ic_0(as_offset(Koff, ikidx(E2EC[0])))
        - theta_v_ic_0(as_offset(Koff, ikidx(E2EC[0]) + 1.0))
        * inv_ddqz_z_full_0(as_offset(Koff, ikidx(E2EC[0])))
    )

    z_theta2 = (
        theta_v_1(as_offset(Koff, ikidx(E2EC[1])))
        + zdiff_gradp(E2EC[1]) * theta_v_ic_1(as_offset(Koff, ikidx(E2EC[1])))
        - theta_v_ic_1(as_offset(Koff, ikidx(E2EC[1]) + 1.0))
        * inv_ddqz_z_full_1(as_offset(Koff, ikidx(E2EC[1])))
    )

    z_hydro_corr = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta2 - z_theta1)
        * 4.0
        / ((z_theta1 + z_theta2) ** 2)
    )
    return z_hydro_corr


@program
def mo_solve_nonhydro_stencil_21(
    theta_v: Field[[CellDim, KDim], float],
    ikidx: Field[[ECDim, KDim], float],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    grav_o_cpd: float,
):
    _mo_solve_nonhydro_stencil_21(
        theta_v,
        ikidx,
        zdiff_gradp,
        theta_v_ic,
        inv_ddqz_z_full,
        inv_dual_edge_length,
        grav_o_cpd,
        out=z_hydro_corr,
    )
