# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestCalculateNabla2AndSmagCoefficientsForVn(StencilTest):
    PROGRAM = calculate_nabla2_and_smag_coefficients_for_vn
    OUTPUTS = ("kh_smag_e", "kh_smag_ec", "z_nabla2_e")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        diff_multfac_smag: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        vn: np.ndarray,
        smag_limit: np.ndarray,
        smag_offset,
        **kwargs,
    ) -> dict:
        e2c2v = connectivities[dims.E2C2VDim]
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
        if np.any(grid.connectivities[dims.E2C2VDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        smag_offset = vpfloat("9.0")
        diff_multfac_smag = data_alloc.random_field(grid, dims.KDim, dtype=vpfloat)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        smag_limit = data_alloc.random_field(grid, dims.KDim, dtype=vpfloat)
        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=wpfloat)

        primal_normal_vert_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
        )
        primal_normal_vert_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
        )
        dual_normal_vert_x = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
        )
        dual_normal_vert_y = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=wpfloat
        )

        primal_normal_vert_x_new = data_alloc.as_1D_sparse_field(primal_normal_vert_x, dims.ECVDim)
        primal_normal_vert_y_new = data_alloc.as_1D_sparse_field(primal_normal_vert_y, dims.ECVDim)
        dual_normal_vert_x_new = data_alloc.as_1D_sparse_field(dual_normal_vert_x, dims.ECVDim)
        dual_normal_vert_y_new = data_alloc.as_1D_sparse_field(dual_normal_vert_y, dims.ECVDim)

        z_nabla2_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        kh_smag_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        kh_smag_ec = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

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
