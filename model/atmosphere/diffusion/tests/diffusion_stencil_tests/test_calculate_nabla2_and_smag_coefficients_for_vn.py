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
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common.dimension import E2C2VDim, ECVDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestCalculateNabla2AndSmagCoefficientsForVn(StencilTest):
    PROGRAM = calculate_nabla2_and_smag_coefficients_for_vn
    OUTPUTS = ("kh_smag_e", "kh_smag_ec", "z_nabla2_e")

    @staticmethod
    def reference(
        grid,
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
        **kwargs,
    ) -> tuple[np.array]:
        e2c2v = grid.connectivities[E2C2VDim]
        primal_normal_vert_x = primal_normal_vert_x.reshape(e2c2v.shape)
        primal_normal_vert_y = primal_normal_vert_y.reshape(e2c2v.shape)
        dual_normal_vert_x = dual_normal_vert_x.reshape(e2c2v.shape)
        dual_normal_vert_y = dual_normal_vert_y.reshape(e2c2v.shape)

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
        ) * (inv_primal_edge_length * inv_primal_edge_length)

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
        ) * (inv_vert_vert_length * inv_vert_vert_length)

        z_nabla2_e = 4.0 * z_nabla2_e

        kh_smag_ec = kh_smag_e
        kh_smag_e = np.maximum(0.0, kh_smag_e - smag_offset)
        kh_smag_e = np.minimum(kh_smag_e, smag_limit)

        return dict(kh_smag_e=kh_smag_e, kh_smag_ec=kh_smag_ec, z_nabla2_e=z_nabla2_e)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[E2C2VDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        u_vert = random_field(grid, VertexDim, KDim, dtype=vpfloat)
        v_vert = random_field(grid, VertexDim, KDim, dtype=vpfloat)
        smag_offset = vpfloat("9.0")
        diff_multfac_smag = random_field(grid, KDim, dtype=vpfloat)
        tangent_orientation = random_field(grid, EdgeDim, dtype=wpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        smag_limit = random_field(grid, KDim, dtype=vpfloat)
        inv_vert_vert_length = random_field(grid, EdgeDim, dtype=wpfloat)
        inv_primal_edge_length = random_field(grid, EdgeDim, dtype=wpfloat)

        primal_normal_vert_x = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)
        primal_normal_vert_y = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)
        dual_normal_vert_x = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)
        dual_normal_vert_y = random_field(grid, EdgeDim, E2C2VDim, dtype=wpfloat)

        primal_normal_vert_x_new = as_1D_sparse_field(primal_normal_vert_x, ECVDim)
        primal_normal_vert_y_new = as_1D_sparse_field(primal_normal_vert_y, ECVDim)
        dual_normal_vert_x_new = as_1D_sparse_field(dual_normal_vert_x, ECVDim)
        dual_normal_vert_y_new = as_1D_sparse_field(dual_normal_vert_y, ECVDim)

        z_nabla2_e = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)
        kh_smag_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        kh_smag_ec = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            diff_multfac_smag=diff_multfac_smag,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_x=primal_normal_vert_x_new,
            primal_normal_vert_y=primal_normal_vert_y_new,
            dual_normal_vert_x=dual_normal_vert_x_new,
            dual_normal_vert_y=dual_normal_vert_y_new,
            vn=vn,
            smag_limit=smag_limit,
            kh_smag_e=kh_smag_e,
            kh_smag_ec=kh_smag_ec,
            z_nabla2_e=z_nabla2_e,
            smag_offset=smag_offset,
            horizontal_start=0,
            horizontal_end=grid.num_edges,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
