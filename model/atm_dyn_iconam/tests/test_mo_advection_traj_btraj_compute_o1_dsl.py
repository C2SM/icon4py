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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atm_dyn_iconam.mo_advection_traj_btraj_compute_o1_dsl import (
    mo_advection_traj_btraj_compute_o1_dsl,
)
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim

from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, constant_field, random_field
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_advection_traj_btraj_compute_o1_dsl_numpy(
    p_vn: np.array,
    p_vt: np.array,
    cell_idx: np.array,
    cell_blk: np.array,
    pos_on_tplane_e_1: np.array,
    pos_on_tplane_e_2: np.array,
    primal_normal_cell_1: np.array,
    dual_normal_cell_1: np.array,
    primal_normal_cell_2: np.array,
    dual_normal_cell_2: np.array,
    p_dthalf: float,
) -> np.array:
    lvn_pos = np.where(p_vn > 0.0, True, False)
    cell_idx = np.expand_dims(cell_idx, axis=-1)
    cell_blk = np.expand_dims(cell_blk, axis=-1)
    pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
    pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
    primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
    dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
    primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
    dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

    p_cell_idx = np.where(lvn_pos, cell_idx[:, 0], cell_idx[:, 1])
    p_cell_blk = np.where(lvn_pos, cell_blk[:, 0], cell_blk[:, 1])

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

    return p_cell_idx, p_cell_blk, p_distv_bary_1, p_distv_bary_2


def test_mo_advection_traj_btraj_compute_o1_dsl():
    mesh = SimpleMesh()

    p_vn = random_field(mesh, EdgeDim, KDim)
    p_vt = random_field(mesh, EdgeDim, KDim)
    cell_idx = np.asarray(mesh.e2c, dtype=int32)
    cell_idx_new = as_1D_sparse_field(cell_idx, ECDim)
    cell_blk = constant_field(mesh, 1, EdgeDim, E2CDim, dtype=int32)
    cell_blk_new = as_1D_sparse_field(cell_blk, ECDim)
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
    p_cell_idx = constant_field(mesh, 0, EdgeDim, KDim, dtype=int32)
    p_cell_blk = constant_field(mesh, 0, EdgeDim, KDim, dtype=int32)
    p_distv_bary_1 = random_field(mesh, EdgeDim, KDim)
    p_distv_bary_2 = random_field(mesh, EdgeDim, KDim)
    p_dthalf = 2.0

    (
        p_cell_idx_ref,
        p_cell_blk_ref,
        p_distv_bary_1_ref,
        p_distv_bary_2_ref,
    ) = mo_advection_traj_btraj_compute_o1_dsl_numpy(
        np.asarray(p_vn),
        np.asarray(p_vt),
        np.asarray(cell_idx),
        np.asarray(cell_blk),
        np.asarray(pos_on_tplane_e_1),
        np.asarray(pos_on_tplane_e_2),
        np.asarray(primal_normal_cell_1),
        np.asarray(dual_normal_cell_1),
        np.asarray(primal_normal_cell_2),
        np.asarray(dual_normal_cell_2),
        p_dthalf,
    )

    mo_advection_traj_btraj_compute_o1_dsl(
        p_vn,
        p_vt,
        cell_idx_new,
        cell_blk_new,
        pos_on_tplane_e_1_new,
        pos_on_tplane_e_2_new,
        primal_normal_cell_1_new,
        dual_normal_cell_1_new,
        primal_normal_cell_2_new,
        dual_normal_cell_2_new,
        p_cell_idx,
        p_cell_blk,
        p_distv_bary_1,
        p_distv_bary_2,
        p_dthalf,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
        },
    )
    assert np.allclose(p_cell_idx, p_cell_idx_ref)
    assert np.allclose(p_cell_blk, p_cell_blk_ref)
    assert np.allclose(p_distv_bary_1, p_distv_bary_1_ref)
    assert np.allclose(p_distv_bary_2, p_distv_bary_2_ref)
