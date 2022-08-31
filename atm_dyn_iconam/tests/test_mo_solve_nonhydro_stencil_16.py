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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_16 import (
    mo_solve_nonhydro_stencil_16,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


def mo_solve_nonhydro_stencil_16_numpy(
    e2c: np.array,
    p_vn: np.array,
    rho_ref_me: np.array,
    theta_ref_me: np.array,
    p_distv_bary_1: np.array,
    p_distv_bary_2: np.array,
    z_grad_rth_1: np.array,
    z_grad_rth_2: np.array,
    z_grad_rth_3: np.array,
    z_grad_rth_4: np.array,
    z_rth_pr_1: np.array,
    z_rth_pr_2: np.array,
) -> np.array:

    z_rth_pr_1_e2c = z_rth_pr_1[e2c]
    z_rth_pr_2_e2c = z_rth_pr_2[e2c]
    z_grad_rth_1_e2c = z_grad_rth_1[e2c]
    z_grad_rth_2_e2c = z_grad_rth_2[e2c]
    z_grad_rth_3_e2c = z_grad_rth_3[e2c]
    z_grad_rth_4_e2c = z_grad_rth_4[e2c]

    z_rho_e = np.where(
        p_vn > 0,
        rho_ref_me
        + z_rth_pr_1_e2c[:, 0]
        + p_distv_bary_1 * z_grad_rth_1_e2c[:, 0]
        + p_distv_bary_2 * z_grad_rth_2_e2c[:, 0],
        rho_ref_me
        + z_rth_pr_1_e2c[:, 1]
        + p_distv_bary_1 * z_grad_rth_1_e2c[:, 1]
        + p_distv_bary_2 * z_grad_rth_2_e2c[:, 1],
    )

    z_theta_v_e = np.where(
        p_vn > 0,
        theta_ref_me
        + z_rth_pr_2_e2c[:, 0]
        + p_distv_bary_1 * z_grad_rth_3_e2c[:, 0]
        + p_distv_bary_2 * z_grad_rth_4_e2c[:, 0],
        theta_ref_me
        + z_rth_pr_2_e2c[:, 1]
        + p_distv_bary_1 * z_grad_rth_3_e2c[:, 1]
        + p_distv_bary_2 * z_grad_rth_4_e2c[:, 1],
    )

    return z_rho_e, z_theta_v_e


def test_mo_solve_nonhydro_stencil_16():
    mesh = SimpleMesh()

    p_vn = random_field(mesh, EdgeDim, KDim)
    rho_ref_me = random_field(mesh, EdgeDim, KDim)
    theta_ref_me = random_field(mesh, EdgeDim, KDim)
    p_distv_bary_1 = random_field(mesh, EdgeDim, KDim)
    p_distv_bary_2 = random_field(mesh, EdgeDim, KDim)
    z_grad_rth_1 = random_field(mesh, CellDim, KDim)
    z_grad_rth_2 = random_field(mesh, CellDim, KDim)
    z_grad_rth_3 = random_field(mesh, CellDim, KDim)
    z_grad_rth_4 = random_field(mesh, CellDim, KDim)
    z_rth_pr_1 = random_field(mesh, CellDim, KDim)
    z_rth_pr_2 = random_field(mesh, CellDim, KDim)
    z_rho_e = random_field(mesh, EdgeDim, KDim)
    z_theta_v_e = random_field(mesh, EdgeDim, KDim)

    z_rho_e_ref, z_theta_v_e_ref = mo_solve_nonhydro_stencil_16_numpy(
        mesh.e2c,
        np.asarray(p_vn),
        np.asarray(rho_ref_me),
        np.asarray(theta_ref_me),
        np.asarray(p_distv_bary_1),
        np.asarray(p_distv_bary_2),
        np.asarray(z_grad_rth_1),
        np.asarray(z_grad_rth_2),
        np.asarray(z_grad_rth_3),
        np.asarray(z_grad_rth_4),
        np.asarray(z_rth_pr_1),
        np.asarray(z_rth_pr_2),
    )

    mo_solve_nonhydro_stencil_16(
        p_vn,
        rho_ref_me,
        theta_ref_me,
        p_distv_bary_1,
        p_distv_bary_2,
        z_grad_rth_1,
        z_grad_rth_2,
        z_grad_rth_3,
        z_grad_rth_4,
        z_rth_pr_1,
        z_rth_pr_2,
        z_rho_e,
        z_theta_v_e,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
        },
    )
    assert np.allclose(z_rho_e, z_rho_e_ref)
    assert np.allclose(z_theta_v_e, z_theta_v_e_ref)
