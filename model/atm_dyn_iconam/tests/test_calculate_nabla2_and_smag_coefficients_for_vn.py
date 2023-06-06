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

from icon4py.model.atm_dyn_iconam.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common.dimension import (
    E2C2VDim,
    ECVDim,
    EdgeDim,
    KDim,
    VertexDim,
)

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def calculate_nabla2_and_smag_coefficients_for_vn_numpy(
    e2c2v: np.array,
    diff_multfac_smag: np.array,
    tangent_orientation: np.array,
    inv_primal_edge_length: np.array,
    inv_vert_vert_length: np.array,
    u_vert: np.array,
    v_vert: np.array,
    primal_normal_vert_x: np.array,
    primal_normal_vert_y: np.array,
    dual_normal_vert_x: np.array,
    dual_normal_vert_y: np.array,
    vn: np.array,
    smag_limit: np.array,
    smag_offset,
) -> tuple[np.array]:
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]
    dual_normal_vert_x = np.expand_dims(dual_normal_vert_x, axis=-1)
    dual_normal_vert_y = np.expand_dims(dual_normal_vert_y, axis=-1)
    primal_normal_vert_x = np.expand_dims(primal_normal_vert_x, axis=-1)
    primal_normal_vert_y = np.expand_dims(primal_normal_vert_y, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    dvt_tang = (
        -(
            u_vert_e2c2v[:, 0] * dual_normal_vert_x[:, 0]
            + v_vert_e2c2v[:, 0] * dual_normal_vert_y[:, 0]
        )
    ) + (
        u_vert_e2c2v[:, 1] * dual_normal_vert_x[:, 1]
        + v_vert_e2c2v[:, 1] * dual_normal_vert_y[:, 1]
    )

    dvt_norm = (
        -(
            u_vert_e2c2v[:, 2] * dual_normal_vert_x[:, 2]
            + v_vert_e2c2v[:, 2] * dual_normal_vert_y[:, 2]
        )
    ) + (
        u_vert_e2c2v[:, 3] * dual_normal_vert_x[:, 3]
        + v_vert_e2c2v[:, 3] * dual_normal_vert_y[:, 3]
    )

    kh_smag_1 = (
        -(
            u_vert_e2c2v[:, 0] * primal_normal_vert_x[:, 0]
            + v_vert_e2c2v[:, 0] * primal_normal_vert_y[:, 0]
        )
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_x[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_y[:, 1]
    )

    dvt_tang = dvt_tang * tangent_orientation

    kh_smag_1 = (kh_smag_1 * tangent_orientation * inv_primal_edge_length) + (
        dvt_norm * inv_vert_vert_length
    )

    kh_smag_1 = kh_smag_1 * kh_smag_1

    kh_smag_2 = (
        -(
            u_vert_e2c2v[:, 2] * primal_normal_vert_x[:, 2]
            + v_vert_e2c2v[:, 2] * primal_normal_vert_y[:, 2]
        )
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_x[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_y[:, 3]
    )

    kh_smag_2 = (kh_smag_2 * inv_vert_vert_length) - (dvt_tang * inv_primal_edge_length)

    kh_smag_2 = kh_smag_2 * kh_smag_2

    kh_smag_e = diff_multfac_smag * np.sqrt(kh_smag_2 + kh_smag_1)

    z_nabla2_e = (
        (
            (
                u_vert_e2c2v[:, 0] * primal_normal_vert_x[:, 0]
                + v_vert_e2c2v[:, 0] * primal_normal_vert_y[:, 0]
            )
            + (
                u_vert_e2c2v[:, 1] * primal_normal_vert_x[:, 1]
                + v_vert_e2c2v[:, 1] * primal_normal_vert_y[:, 1]
            )
        )
        - 2.0 * vn
    ) * (inv_primal_edge_length**2)

    z_nabla2_e = z_nabla2_e + (
        (
            (
                u_vert_e2c2v[:, 2] * primal_normal_vert_x[:, 2]
                + v_vert_e2c2v[:, 2] * primal_normal_vert_y[:, 2]
            )
            + (
                u_vert_e2c2v[:, 3] * primal_normal_vert_x[:, 3]
                + v_vert_e2c2v[:, 3] * primal_normal_vert_y[:, 3]
            )
        )
        - 2.0 * vn
    ) * (inv_vert_vert_length**2)

    z_nabla2_e = 4.0 * z_nabla2_e

    kh_smag_ec = kh_smag_e
    kh_smag_e = np.maximum(0.0, kh_smag_e - smag_offset)
    kh_smag_e = np.minimum(kh_smag_e, smag_limit)

    return kh_smag_e, kh_smag_ec, z_nabla2_e


def test_calculate_nabla2_and_smag_coefficients_for_vn():
    mesh = SimpleMesh()

    u_vert = random_field(mesh, VertexDim, KDim)
    v_vert = random_field(mesh, VertexDim, KDim)
    smag_offset = 9.0
    diff_multfac_smag = random_field(mesh, KDim)
    tangent_orientation = random_field(mesh, EdgeDim)
    vn = random_field(mesh, EdgeDim, KDim)
    smag_limit = random_field(mesh, KDim)
    inv_vert_vert_length = random_field(mesh, EdgeDim)
    inv_primal_edge_length = random_field(mesh, EdgeDim)

    primal_normal_vert_x = random_field(mesh, EdgeDim, E2C2VDim)
    primal_normal_vert_y = random_field(mesh, EdgeDim, E2C2VDim)
    dual_normal_vert_x = random_field(mesh, EdgeDim, E2C2VDim)
    dual_normal_vert_y = random_field(mesh, EdgeDim, E2C2VDim)

    primal_normal_vert_x_new = as_1D_sparse_field(primal_normal_vert_x, ECVDim)
    primal_normal_vert_y_new = as_1D_sparse_field(primal_normal_vert_y, ECVDim)
    dual_normal_vert_x_new = as_1D_sparse_field(dual_normal_vert_x, ECVDim)
    dual_normal_vert_y_new = as_1D_sparse_field(dual_normal_vert_y, ECVDim)

    z_nabla2_e = zero_field(mesh, EdgeDim, KDim)
    kh_smag_e = zero_field(mesh, EdgeDim, KDim)
    kh_smag_ec = zero_field(mesh, EdgeDim, KDim)

    (
        kh_smag_e_ref,
        kh_smag_ec_ref,
        z_nabla2_e_ref,
    ) = calculate_nabla2_and_smag_coefficients_for_vn_numpy(
        mesh.e2c2v,
        np.asarray(diff_multfac_smag),
        np.asarray(tangent_orientation),
        np.asarray(inv_primal_edge_length),
        np.asarray(inv_vert_vert_length),
        np.asarray(u_vert),
        np.asarray(v_vert),
        np.asarray(primal_normal_vert_x),
        np.asarray(primal_normal_vert_y),
        np.asarray(dual_normal_vert_x),
        np.asarray(dual_normal_vert_y),
        np.asarray(vn),
        np.asarray(smag_limit),
        smag_offset,
    )

    calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert,
        v_vert,
        primal_normal_vert_x_new,
        primal_normal_vert_y_new,
        dual_normal_vert_x_new,
        dual_normal_vert_y_new,
        vn,
        smag_limit,
        kh_smag_e,
        kh_smag_ec,
        z_nabla2_e,
        smag_offset,
        0,
        mesh.n_edges,
        0,
        mesh.k_level,
        offset_provider={
            "E2C2V": mesh.get_e2c2v_offset_provider(),
            "E2ECV": StridedNeighborOffsetProvider(EdgeDim, ECVDim, mesh.n_e2c2v),
        },
    )
    assert np.allclose(kh_smag_e_ref, kh_smag_e)
    assert np.allclose(kh_smag_ec_ref, kh_smag_ec)
    assert np.allclose(z_nabla2_e_ref, z_nabla2_e)
