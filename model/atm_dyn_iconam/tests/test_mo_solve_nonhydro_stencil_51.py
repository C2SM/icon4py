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

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_51 import (
    mo_solve_nonhydro_stencil_51,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_51_z_q_numpy(
    z_c: np.array,
    z_b: np.array,
) -> np.array:
    return -z_c / z_b


def mo_solve_nonhydro_stencil_51_w_nnew_numpy(
    z_gamma: np.array,
    z_b: np.array,
    z_w_expl: np.array,
    z_exner_expl: np.array,
) -> np.array:
    z_exner_expl_k_minus_1 = np.roll(z_exner_expl, shift=1, axis=1)
    w_nnew = z_w_expl[:, :-1] - z_gamma * (z_exner_expl_k_minus_1 - z_exner_expl)
    return w_nnew / z_b


def mo_solve_nonhydro_stencil_51_numpy(
    vwind_impl_wgt: np.array,
    theta_v_ic: np.array,
    ddqz_z_half: np.array,
    z_beta: np.array,
    z_alpha: np.array,
    z_w_expl: np.array,
    z_exner_expl: np.array,
    dtime: float,
    cpd: float,
) -> tuple[np.array]:
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_alpha_k_plus_1 = z_alpha[:, 1:]
    z_beta_k_minus_1 = np.roll(z_beta, shift=1, axis=1)
    z_c = -z_gamma * z_beta * z_alpha_k_plus_1
    z_b = 1.0 + z_gamma * z_alpha[:, :-1] * (z_beta_k_minus_1 + z_beta)
    z_q = mo_solve_nonhydro_stencil_51_z_q_numpy(z_c, z_b)
    w_nnew = mo_solve_nonhydro_stencil_51_w_nnew_numpy(
        z_gamma, z_b, z_w_expl, z_exner_expl
    )

    return z_q, w_nnew


def test_mo_solve_nonhydro_stencil_51():
    mesh = SimpleMesh()
    z_q = random_field(mesh, CellDim, KDim)
    w_nnew = random_field(mesh, CellDim, KDim)
    vwind_impl_wgt = random_field(mesh, CellDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    ddqz_z_half = random_field(mesh, CellDim, KDim, low=0.5, high=1.5)
    z_beta = random_field(mesh, CellDim, KDim, low=0.5, high=1.5)
    z_alpha = random_field(mesh, CellDim, KDim, low=0.5, high=1.5, extend={KDim: 1})
    z_w_expl = random_field(mesh, CellDim, KDim, extend={KDim: 1})
    z_exner_expl = random_field(mesh, CellDim, KDim)
    dtime = 10.0
    cpd = 1.0

    z_q_ref, w_nnew_ref = mo_solve_nonhydro_stencil_51_numpy(
        np.asarray(vwind_impl_wgt),
        np.asarray(theta_v_ic),
        np.asarray(ddqz_z_half),
        np.asarray(z_beta),
        np.asarray(z_alpha),
        np.asarray(z_w_expl),
        np.asarray(z_exner_expl),
        dtime,
        cpd,
    )

    mo_solve_nonhydro_stencil_51(
        z_q,
        w_nnew,
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_beta,
        z_alpha,
        z_w_expl,
        z_exner_expl,
        dtime,
        cpd,
        offset_provider={"Koff": KDim},
    )  # TODO passing `w` as in and out is not guaranteed to work

    assert np.allclose(z_q_ref[:, 1:], z_q[:, 1:])
    assert np.allclose(w_nnew_ref[:, 1:], w_nnew[:, 1:])
