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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_04 import (
    mo_nh_diffusion_stencil_04,
)
from icon4py.common.dimension import VertexDim, EdgeDim, ECVDim, E2C2VDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field
from functional.iterator.embedded import FelixOffsetProvider


def mo_nh_diffusion_stencil_04_numpy(
    e2c2v: np.array,
    u_vert: np.array,
    v_vert: np.array,
    primal_normal_vert_v1: np.array,
    primal_normal_vert_v2: np.array,
    z_nabla2_e: np.array,
    inv_vert_vert_length: np.array,
    inv_primal_edge_length: np.array,
) -> np.array:
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]
    primal_normal_vert_v1 = np.expand_dims(primal_normal_vert_v1, axis=-1)
    primal_normal_vert_v2 = np.expand_dims(primal_normal_vert_v2, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    nabv_tang = (
        u_vert_e2c2v[:, 0] * primal_normal_vert_v1[:, 0]
        + v_vert_e2c2v[:, 0] * primal_normal_vert_v2[:, 0]
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_v1[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_v2[:, 1]
    )
    nabv_norm = (
        u_vert_e2c2v[:, 2] * primal_normal_vert_v1[:, 2]
        + v_vert_e2c2v[:, 2] * primal_normal_vert_v2[:, 2]
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_v1[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_v2[:, 3]
    )
    z_nabla4_e2 = 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length**2
        + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length**2
    )
    return z_nabla4_e2


def test_mo_nh_diffusion_stencil_04():
    mesh = SimpleMesh()

    u_vert = random_field(mesh, VertexDim, KDim)
    v_vert = random_field(mesh, VertexDim, KDim)

    primal_normal_vert_v1 = random_field(mesh, EdgeDim, E2C2VDim)
    primal_normal_vert_v2 = random_field(mesh, EdgeDim, E2C2VDim)

    primal_normal_vert_v1_new = random_field(mesh, ECVDim)
    primal_normal_vert_v2_new = random_field(mesh, ECVDim)

    # todo: do this properly
    for i in range(0, mesh.n_edges):
        for j in range(0, mesh.n_e2c2v):
            primal_normal_vert_v1_new[i * mesh.n_e2c2v + j] = primal_normal_vert_v1[
                i, j
            ]
            primal_normal_vert_v2_new[i * mesh.n_e2c2v + j] = primal_normal_vert_v2[
                i, j
            ]

    z_nabla2_e = random_field(mesh, EdgeDim, KDim)
    inv_vert_vert_length = random_field(mesh, EdgeDim)
    inv_primal_edge_length = random_field(mesh, EdgeDim)

    z_nabla4_e2 = zero_field(mesh, EdgeDim, KDim)

    z_nabla4_e2_ref = mo_nh_diffusion_stencil_04_numpy(
        mesh.e2c2v,
        np.asarray(u_vert),
        np.asarray(v_vert),
        np.asarray(primal_normal_vert_v1),
        np.asarray(primal_normal_vert_v2),
        np.asarray(z_nabla2_e),
        np.asarray(inv_vert_vert_length),
        np.asarray(inv_primal_edge_length),
    )

    mo_nh_diffusion_stencil_04(
        u_vert,
        v_vert,
        primal_normal_vert_v1_new,
        primal_normal_vert_v2_new,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        z_nabla4_e2,
        offset_provider={
            "E2C2V": mesh.get_e2c2v_offset_provider(),
            "E2ECV": FelixOffsetProvider(
                (mesh.n_edges, mesh.n_e2c2v), EdgeDim, ECVDim, mesh.n_e2c2v
            ),
        },
    )

    assert np.allclose(z_nabla4_e2, z_nabla4_e2_ref)


if __name__ == "__main__":
    test_mo_nh_diffusion_stencil_04()
