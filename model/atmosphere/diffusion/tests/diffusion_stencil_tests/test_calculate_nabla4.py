# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import calculate_nabla4
from icon4py.model.common.dimension import E2C2VDim, ECVDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


def calculate_nabla4_numpy(
    grid,
    u_vert: np.array,
    v_vert: np.array,
    primal_normal_vert_v1: np.array,
    primal_normal_vert_v2: np.array,
    z_nabla2_e: np.array,
    inv_vert_vert_length: np.array,
    inv_primal_edge_length: np.array,
) -> np.array:
    e2c2v = grid.connectivities[E2C2VDim]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]

    primal_normal_vert_v1 = primal_normal_vert_v1.reshape(e2c2v.shape)
    primal_normal_vert_v2 = primal_normal_vert_v2.reshape(e2c2v.shape)

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


class TestCalculateNabla4(StencilTest):
    PROGRAM = calculate_nabla4
    OUTPUTS = ("z_nabla4_e2",)

    @staticmethod
    def reference(
        grid,
        u_vert: np.array,
        v_vert: np.array,
        primal_normal_vert_v1: np.array,
        primal_normal_vert_v2: np.array,
        z_nabla2_e: np.array,
        inv_vert_vert_length: np.array,
        inv_primal_edge_length: np.array,
        **kwargs,
    ) -> dict:
        z_nabla4_e2 = calculate_nabla4_numpy(
            grid,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )
        return dict(z_nabla4_e2=z_nabla4_e2)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[E2C2VDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        u_vert = random_field(grid, VertexDim, KDim, dtype=vpfloat)
        v_vert = random_field(grid, VertexDim, KDim, dtype=vpfloat)

        primal_normal_vert_v1 = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)
        primal_normal_vert_v2 = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)

        primal_normal_vert_v1_new = as_1D_sparse_field(primal_normal_vert_v1, ECVDim)
        primal_normal_vert_v2_new = as_1D_sparse_field(primal_normal_vert_v2, ECVDim)

        z_nabla2_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        inv_vert_vert_length = random_field(grid, EdgeDim, dtype=wpfloat)
        inv_primal_edge_length = random_field(grid, EdgeDim, dtype=wpfloat)

        z_nabla4_e2 = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1_new,
            primal_normal_vert_v2=primal_normal_vert_v2_new,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            z_nabla4_e2=z_nabla4_e2,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
