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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_07 import (
    mo_velocity_advection_stencil_07,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_velocity_advection_stencil_07_numpy(
    nlev: int,
    num_edge: int,
    e2c: np.array,
    e2v: np.array,
    vn_ie: np.array,
    inv_dual_edge_length: np.array,
    w: np.array,
    z_vt_ie: np.array,
    inv_primal_edge_length: np.array,
    tangent_orientation: np.array,
    z_w_v: np.array,
):
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    red_w = np.zeros((num_edge, nlev))
    red_z_w_v = np.zeros((num_edge, nlev))
    red_w = w[e2c][:, 0] - w[e2c][:, 1]
    red_z_w_v = z_w_v[e2v][:, 0] - z_w_v[e2v][:, 1]
    
    z_v_grad_w = (
        vn_ie * inv_dual_edge_length * red_w
        + z_vt_ie * inv_primal_edge_length * tangent_orientation * red_z_w_v
    )
    return z_v_grad_w


def test_mo_velocity_advection_stencil_07():
    mesh = SimpleMesh()

    vn_ie = random_field(mesh, EdgeDim, KDim)
    inv_dual_edge_length = random_field(mesh, EdgeDim)
    w = random_field(mesh, CellDim, KDim)
    z_vt_ie = random_field(mesh, EdgeDim, KDim)
    inv_primal_edge_length = random_field(mesh, EdgeDim)
    tangent_orientation = random_field(mesh, EdgeDim)
    z_w_v = random_field(mesh, VertexDim, KDim)
    z_v_grad_w = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_07_numpy(
        mesh.k_level,
        mesh.n_edges,
        mesh.e2c,
        mesh.e2v,
        np.asarray(vn_ie),
        np.asarray(inv_dual_edge_length),
        np.asarray(w),
        np.asarray(z_vt_ie),
        np.asarray(inv_primal_edge_length),
        np.asarray(tangent_orientation),
        np.asarray(z_w_v),
    )
    mo_velocity_advection_stencil_07(
        vn_ie,
        inv_dual_edge_length,
        w,
        z_vt_ie,
        inv_primal_edge_length,
        tangent_orientation,
        z_w_v,
        z_v_grad_w,
        offset_provider={
            "E2C": mesh.get_e2c_offset_provider(),
            "E2V": mesh.get_e2v_offset_provider(),
        },
    )
    assert np.allclose(z_v_grad_w, ref)
