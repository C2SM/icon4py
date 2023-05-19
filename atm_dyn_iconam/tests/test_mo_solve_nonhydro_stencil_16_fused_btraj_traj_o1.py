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
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider
from simple_mesh import SimpleMesh
from utils import as_1D_sparse_field, random_field

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
)
from icon4py.common.dimension import CellDim, E2CDim, ECDim, EdgeDim, KDim


def compute_btraj_numpy(
    p_vn: np.array,
    p_vt: np.array,
    pos_on_tplane_e_1: np.array,
    pos_on_tplane_e_2: np.array,
    primal_normal_cell_1: np.array,
    dual_normal_cell_1: np.array,
    primal_normal_cell_2: np.array,
    dual_normal_cell_2: np.array,
    p_dthalf: float,
) -> np.array:
    lvn_pos = np.where(p_vn > 0.0, True, False)
    pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
    pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
    primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
    dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
    primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
    dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

    z_ntdistv_bary_1 = -(
        p_vn * p_dthalf
        + np.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
    )
    z_ntdistv_bary_2 = -(
        p_vt * p_dthalf
        + np.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
    )

    p_distv_bary_1 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1[:, 0]
        + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_1[:, 1]
        + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
    )

    p_distv_bary_2 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2[:, 0]
        + z_ntdistv_bary_2 * dual_normal_cell_2[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_2[:, 1]
        + z_ntdistv_bary_2 * dual_normal_cell_2[:, 1],
    )

    return p_distv_bary_1, p_distv_bary_2


def sten_16_numpy(
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


def mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_numpy(
    e2c: np.array,
    p_vn: np.array,
    p_vt: np.array,
    pos_on_tplane_e_1: np.array,
    pos_on_tplane_e_2: np.array,
    primal_normal_cell_1: np.array,
    dual_normal_cell_1: np.array,
    primal_normal_cell_2: np.array,
    dual_normal_cell_2: np.array,
    p_dthalf: float,
    rho_ref_me: np.array,
    theta_ref_me: np.array,
    z_grad_rth_1: np.array,
    z_grad_rth_2: np.array,
    z_grad_rth_3: np.array,
    z_grad_rth_4: np.array,
    z_rth_pr_1: np.array,
    z_rth_pr_2: np.array,
):
    p_distv_bary_1, p_distv_bary_2 = compute_btraj_numpy(
        p_vn,
        p_vt,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        p_dthalf,
    )

    z_rho_e, z_theta_v_e = sten_16_numpy(
        e2c,
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
    )

    return z_rho_e, z_theta_v_e


def test_mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1():
    mesh = SimpleMesh()

    p_vn = random_field(mesh, EdgeDim, KDim)
    p_vt = random_field(mesh, EdgeDim, KDim)
    pos_on_tplane_e_1 = random_field(mesh, EdgeDim, E2CDim)
    pos_on_tplane_e_1_new = as_1D_sparse_field(pos_on_tplane_e_1, ECDim)
    pos_on_tplane_e_2 = random_field(mesh, EdgeDim, E2CDim)
    pos_on_tplane_e_2_new = as_1D_sparse_field(pos_on_tplane_e_2, ECDim)
    primal_normal_cell_1 = random_field(mesh, EdgeDim, E2CDim)
    primal_normal_cell_1_new = as_1D_sparse_field(primal_normal_cell_1, ECDim)
    dual_normal_cell_1 = random_field(mesh, EdgeDim, E2CDim)
    dual_normal_cell_1_new = as_1D_sparse_field(dual_normal_cell_1, ECDim)
    primal_normal_cell_2 = random_field(mesh, EdgeDim, E2CDim)
    primal_normal_cell_2_new = as_1D_sparse_field(primal_normal_cell_2, ECDim)
    dual_normal_cell_2 = random_field(mesh, EdgeDim, E2CDim)
    dual_normal_cell_2_new = as_1D_sparse_field(dual_normal_cell_2, ECDim)
    p_dthalf = 2.0

    rho_ref_me = random_field(mesh, EdgeDim, KDim)
    theta_ref_me = random_field(mesh, EdgeDim, KDim)
    z_grad_rth_1 = random_field(mesh, CellDim, KDim)
    z_grad_rth_2 = random_field(mesh, CellDim, KDim)
    z_grad_rth_3 = random_field(mesh, CellDim, KDim)
    z_grad_rth_4 = random_field(mesh, CellDim, KDim)
    z_rth_pr_1 = random_field(mesh, CellDim, KDim)
    z_rth_pr_2 = random_field(mesh, CellDim, KDim)
    z_rho_e = random_field(mesh, EdgeDim, KDim)
    z_theta_v_e = random_field(mesh, EdgeDim, KDim)

    (
        z_rho_e_ref,
        z_theta_v_e_ref,
    ) = mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1_numpy(
        mesh.e2c,
        np.asarray(p_vn),
        np.asarray(p_vt),
        np.asarray(pos_on_tplane_e_1),
        np.asarray(pos_on_tplane_e_2),
        np.asarray(primal_normal_cell_1),
        np.asarray(dual_normal_cell_1),
        np.asarray(primal_normal_cell_2),
        np.asarray(dual_normal_cell_2),
        p_dthalf,
        np.asarray(rho_ref_me),
        np.asarray(theta_ref_me),
        np.asarray(z_grad_rth_1),
        np.asarray(z_grad_rth_2),
        np.asarray(z_grad_rth_3),
        np.asarray(z_grad_rth_4),
        np.asarray(z_rth_pr_1),
        np.asarray(z_rth_pr_2),
    )

    mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
        p_vn,
        p_vt,
        pos_on_tplane_e_1_new,
        pos_on_tplane_e_2_new,
        primal_normal_cell_1_new,
        dual_normal_cell_1_new,
        primal_normal_cell_2_new,
        dual_normal_cell_2_new,
        p_dthalf,
        rho_ref_me,
        theta_ref_me,
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
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
        },
    )
    assert np.allclose(z_rho_e, z_rho_e_ref)
    assert np.allclose(z_theta_v_e, z_theta_v_e_ref)
