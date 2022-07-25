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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_44 import (
    mo_solve_nonhydro_stencil_44,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import (
    broadcasted_field,
    random_field,
    zero_field,
)


# TODO: change dtime, rv, cvd back to type float


def mo_solve_nonhydro_stencil_44_z_beta_numpy(
    exner_nnow: np.array,
    rho_nnow: np.array,
    theta_v_nnow: np.array,
    inv_ddqz_z_full: np.array,
    dtime: np.array,
    rd: np.array,
    cvd: np.array,
) -> np.array:
    z_beta = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full
    return z_beta


def mo_solve_nonhydro_stencil_44_z_alpha_numpy(
    vwind_impl_wgt: np.array, theta_v_ic: np.array, rho_ic: np.array
) -> np.array:
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
    z_alpha = vwind_impl_wgt * theta_v_ic * rho_ic
    return z_alpha


def mo_solve_nonhydro_stencil_44_numpy(
    exner_nnow: np.array,
    rho_nnow: np.array,
    theta_v_nnow: np.array,
    inv_ddqz_z_full: np.array,
    vwind_impl_wgt: np.array,
    theta_v_ic: np.array,
    rho_ic: np.array,
    dtime: np.array,
    rd: np.array,
    cvd: np.array,
) -> tuple[np.array]:
    z_beta = mo_solve_nonhydro_stencil_44_z_beta_numpy(
        exner_nnow, rho_nnow, theta_v_nnow, inv_ddqz_z_full, dtime, rd, cvd
    )
    z_alpha = mo_solve_nonhydro_stencil_44_z_alpha_numpy(
        vwind_impl_wgt, theta_v_ic, rho_ic
    )
    return z_beta, z_alpha


def test_mo_solve_nonhydro_stencil_44_z_alpha():
    mesh = SimpleMesh()

    exner_nnow = random_field(mesh, CellDim, KDim)
    rho_nnow = random_field(mesh, CellDim, KDim)
    theta_v_nnow = random_field(mesh, CellDim, KDim)
    inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
    vwind_impl_wgt = random_field(mesh, CellDim)
    theta_v_ic = random_field(mesh, CellDim, KDim)
    rho_ic = random_field(mesh, CellDim, KDim)
    z_alpha = zero_field(mesh, CellDim, KDim)
    z_beta = zero_field(mesh, CellDim, KDim)
    dtime = broadcasted_field(10.0, mesh, CellDim, KDim)
    rd = broadcasted_field(5.0, mesh, CellDim, KDim)
    cvd = broadcasted_field(3.0, mesh, CellDim, KDim)

    z_beta_ref, z_alpha_ref = mo_solve_nonhydro_stencil_44_numpy(
        np.asarray(exner_nnow),
        np.asarray(rho_nnow),
        np.asarray(theta_v_nnow),
        np.asarray(inv_ddqz_z_full),
        np.asarray(vwind_impl_wgt),
        np.asarray(theta_v_ic),
        np.asarray(rho_ic),
        np.asarray(dtime),
        np.asarray(rd),
        np.asarray(cvd),
    )

    mo_solve_nonhydro_stencil_44(
        z_beta,
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        z_alpha,
        vwind_impl_wgt,
        theta_v_ic,
        rho_ic,
        dtime,
        rd,
        cvd,
        offset_provider={},
    )
    assert np.allclose(z_beta_ref, z_beta)
    assert np.allclose(z_alpha_ref, z_alpha)
