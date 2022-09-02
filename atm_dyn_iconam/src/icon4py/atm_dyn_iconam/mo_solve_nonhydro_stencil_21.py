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

from functional.iterator.builtins import (
    deref,
    named_range,
    shift,
    unstructured_domain,
)
from functional.iterator.runtime import closure, fendef, fundef

from icon4py.common.dimension import E2C, EdgeDim, KDim, Koff


@fundef
def step(i, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full):
    d_ikidx = deref(shift(i)(ikidx))

    d_theta_v = deref(shift(Koff, d_ikidx, E2C, i)(theta_v))
    s_theta_v_ic = shift(Koff, d_ikidx, E2C, i)(theta_v_ic)
    d_theta_v_ic = deref(s_theta_v_ic)
    d_theta_v_ic_p1 = deref(shift(Koff, 1)(s_theta_v_ic))
    d_inv_ddqz_z_full = deref(shift(Koff, d_ikidx, E2C, i)(inv_ddqz_z_full))
    d_zdiff_gradp = deref(shift(i)(zdiff_gradp))

    return (
        d_theta_v + d_zdiff_gradp * (d_theta_v_ic - d_theta_v_ic_p1) * d_inv_ddqz_z_full
    )


@fundef
def _mo_solve_nonhydro_stencil_21(
    theta_v,
    ikidx,
    zdiff_gradp,
    theta_v_ic,
    inv_ddqz_z_full,
    inv_dual_edge_length,
    grav_o_cpd,
):

    z_theta1 = step(0, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full)
    z_theta2 = step(1, theta_v, ikidx, zdiff_gradp, theta_v_ic, inv_ddqz_z_full)
    z_hydro_corr = (
        grav_o_cpd
        * deref(inv_dual_edge_length)
        * (z_theta2 - z_theta1)
        * 4.0
        / ((z_theta1 + z_theta2) ** 2)
    )
    return z_theta1, z_theta2, z_hydro_corr


@fendef
def mo_solve_nonhydro_stencil_21(
    theta_v,
    ikidx,
    zdiff_gradp,
    theta_v_ic,
    inv_ddqz_z_full,
    inv_dual_edge_length,
    grav_o_cpd,
    z_theta1,
    z_theta2,
    z_hydro_corr,
    hstart: int,
    hend: int,
    kstart: int,
    kend: int,
):
    closure(
        unstructured_domain(
            named_range(EdgeDim, hstart, hend), named_range(KDim, kstart, kend)
        ),
        _mo_solve_nonhydro_stencil_21,
        (z_theta1, z_theta2, z_hydro_corr),
        [
            theta_v,
            ikidx,
            zdiff_gradp,
            theta_v_ic,
            inv_ddqz_z_full,
            inv_dual_edge_length,
            grav_o_cpd,
        ],
    )
