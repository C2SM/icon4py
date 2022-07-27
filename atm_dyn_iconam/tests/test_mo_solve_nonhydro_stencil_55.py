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

import numpy as np

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_55 import (
    mo_solve_nonhydro_stencil_55,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_55_numpy(
    z_rho_expl: np.array,
    vwind_impl_wgt: np.array,
    inv_ddqz_z_full: np.array,
    rho_ic: np.array,
    w: np.array,
    z_exner_expl: np.array,
    exner_ref_mc: np.array,
    z_alpha: np.array,
    z_beta: np.array,
    rho_now: np.array,
    theta_v_now: np.array,
    exner_now: np.array,
    dtime,
    cvd_o_rd,
) -> tuple[np.array]:
    rho_ic_offset_0 = np.roll(rho_ic, shift=0, axis=1)
    rho_ic_offset_1 = np.roll(rho_ic, shift=-1, axis=1)
    w_offset_0 = np.roll(w, shift=0, axis=1)
    w_offset_1 = np.roll(w, shift=-1, axis=1)
    z_alpha_offset_0 = np.roll(z_alpha, shift=0, axis=1)
    z_alpha_offset_1 = np.roll(z_alpha, shift=-1, axis=1)
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=1)
    rho_new = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full * (
        rho_ic_offset_0 * w_offset_0 - rho_ic_offset_1 * w_offset_1
    )
    exner_new = (
        z_exner_expl
        + exner_ref_mc
        - z_beta * (z_alpha_offset_0 * w_offset_0 - z_alpha_offset_1 * w_offset_1)
    )
    theta_v_new = (
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - 1.0) * cvd_o_rd + 1.0)
        / rho_new
    )
    return rho_new, exner_new, theta_v_new


def test_mo_solve_nonhydro_stencil_55():
    mesh = SimpleMesh()

    z_rho_expl = random_field(mesh, CellDim, KDim)
    vwind_impl_wgt = random_field(mesh, CellDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    rho_ic = random_field(mesh, CellDim, KDim)
    w = random_field(mesh, CellDim, KDim)
    z_exner_expl = random_field(mesh, CellDim, KDim)
    exner_ref_mc = random_field(mesh, CellDim, KDim)
    z_alpha = random_field(mesh, CellDim, KDim)
    z_beta = random_field(mesh, CellDim, KDim)
    rho_now = random_field(mesh, CellDim, KDim)
    theta_v_now = random_field(mesh, CellDim, KDim)
    exner_now = random_field(mesh, CellDim, KDim)
    rho_new = zero_field(mesh, CellDim, KDim)
    exner_new = zero_field(mesh, CellDim, KDim)
    theta_v_new = zero_field(mesh, CellDim, KDim)
    dtime = 5.0
    cvd_o_rd = 9.0

    rho_new_ref, exner_new_ref, theta_v_new_ref = mo_solve_nonhydro_stencil_55_numpy(
        np.asarray(z_rho_expl),
        np.asarray(vwind_impl_wgt),
        np.asarray(inv_ddqz_z_full),
        np.asarray(rho_ic),
        np.asarray(w),
        np.asarray(z_exner_expl),
        np.asarray(exner_ref_mc),
        np.asarray(z_alpha),
        np.asarray(z_beta),
        np.asarray(rho_now),
        np.asarray(theta_v_now),
        np.asarray(exner_now),
        dtime,
        cvd_o_rd,
    )

    mo_solve_nonhydro_stencil_55(
        z_rho_expl,
        vwind_impl_wgt,
        inv_ddqz_z_full,
        rho_ic,
        w,
        z_exner_expl,
        exner_ref_mc,
        z_alpha,
        z_beta,
        rho_now,
        theta_v_now,
        exner_now,
        rho_new,
        exner_new,
        theta_v_new,
        dtime,
        cvd_o_rd,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(rho_new.__array__()[:, :-1], rho_new_ref[:, :-1])
    assert np.allclose(exner_new.__array__()[:, :-1], exner_new_ref[:, :-1])
    assert np.allclose(theta_v_new.__array__()[:, :-1], theta_v_new_ref[:, :-1])
