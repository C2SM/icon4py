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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_52 import (
    mo_solve_nonhydro_stencil_52,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_52_numpy(
    vwind_impl_wgt: np.array,
    theta_v_ic: np.array,
    ddqz_z_half: np.array,
    z_alpha: np.array,
    z_beta: np.array,
    z_exner_expl: np.array,
    z_w_expl: np.array,
    z_q: np.array,
    dtime,
    cpd,
) -> tuple[np.array]:
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    last_k_level = z_beta.shape[1]
    z_a = np.zeros_like(z_beta)
    z_c = np.zeros_like(z_beta)
    z_g = np.zeros_like(z_beta)
    z_a[:, 1:] = -z_gamma[:, 1:] * z_beta[:, :-1] * z_alpha[:, :-2]
    z_c[:, 1:] = -z_gamma[:, 1:] * z_beta[:, 1:] * z_alpha[:, 2:]
    z_b = 1.0 + z_gamma[:, 1:] * z_alpha[:, 1:-1] * (z_beta[:, :-1] + z_beta[:, 1:])
    z_g[:, 1:] = 1.0 / (z_b + z_a[:, 1:] * z_q[:, :-1])
    z_q = -z_c * z_g
    # w = np.zeros_like(z_w_expl)
    w = np.zeros_like(z_beta)
    # for k in range(2, last_k_level):
    #     z_a[:, k] = -z_gamma[:, k] * z_beta[:, k - 1] * z_alpha[:, k - 1]
    #     z_b[:, k] = 1.0 + z_gamma[:, k] * z_alpha[:, k] * (
    #         z_beta[:, k - 1] + z_beta[:, k]
    #     )
    #     z_g[:, k] = 1.0 / (z_b[:, k] + z_a[:, k] * z_q[:, k - 1])
    # z_c = -z_gamma * z_beta[k] * z_alpha[k + 1]
    # z_q = -z_c * z_g

    for k in range(2, last_k_level):
        w[:, k] = z_w_expl[:, k] - z_gamma[:, k] * (
            z_exner_expl[:, k - 1] - z_exner_expl[:, k]
        )
        w[:, k] = (w[:, k] - z_a[:, k] * w[:, k - 1]) * z_g[:, k]
    return z_q, w


def test_mo_solve_nonhydro_stencil_52():
    mesh = SimpleMesh()
    vwind_impl_wgt = random_field(mesh, CellDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    ddqz_z_half = random_field(mesh, CellDim, KDim)
    z_alpha = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    z_beta = random_field(mesh, CellDim, KDim)
    z_exner_expl = random_field(mesh, CellDim, KDim)
    z_w_expl = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    dtime = 8.0
    cpd = 7.0

    z_q = zero_field(mesh, CellDim, KDim)
    w = zero_field(mesh, CellDim, KDim)

    z_q_ref, w_ref = mo_solve_nonhydro_stencil_52_numpy(
        np.asarray(vwind_impl_wgt),
        np.asarray(theta_v_ic),
        np.asarray(ddqz_z_half),
        np.asarray(z_alpha),
        np.asarray(z_beta),
        np.asarray(z_exner_expl),
        np.asarray(z_w_expl),
        np.asarray(z_q),
        dtime,
        cpd,
    )

    mo_solve_nonhydro_stencil_52(
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
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(z_q_ref[:, 1:], z_q[:, 1:])
    assert np.allclose(w_ref[:, 2:-2], w[:, 2:-2])  # TODO shape
