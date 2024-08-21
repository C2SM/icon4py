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

from icon4py.model.atmosphere.dycore.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_horizontal_advection_term_for_vertical_velocity_numpy(
    grid,
    vn_ie: np.array,
    inv_dual_edge_length: np.array,
    w: np.array,
    z_vt_ie: np.array,
    inv_primal_edge_length: np.array,
    tangent_orientation: np.array,
    z_w_v: np.array,
) -> np.array:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    w_e2c = w[grid.connectivities[dims.E2CDim]]
    z_w_v_e2v = z_w_v[grid.connectivities[dims.E2VDim]]

    red_w = w_e2c[:, 0] - w_e2c[:, 1]
    red_z_w_v = z_w_v_e2v[:, 0] - z_w_v_e2v[:, 1]

    z_v_grad_w = (
        vn_ie * inv_dual_edge_length * red_w
        + z_vt_ie * inv_primal_edge_length * tangent_orientation * red_z_w_v
    )
    return z_v_grad_w


class TestComputeHorizontalAdvectionTermForVerticalVelocity(StencilTest):
    PROGRAM = compute_horizontal_advection_term_for_vertical_velocity
    OUTPUTS = ("z_v_grad_w",)

    @staticmethod
    def reference(
        grid,
        vn_ie: np.array,
        inv_dual_edge_length: np.array,
        w: np.array,
        z_vt_ie: np.array,
        inv_primal_edge_length: np.array,
        tangent_orientation: np.array,
        z_w_v: np.array,
        **kwargs,
    ) -> dict:
        z_v_grad_w = compute_horizontal_advection_term_for_vertical_velocity_numpy(
            grid,
            vn_ie,
            inv_dual_edge_length,
            w,
            z_vt_ie,
            inv_primal_edge_length,
            tangent_orientation,
            z_w_v,
        )
        return dict(z_v_grad_w=z_v_grad_w)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1) or np.any(
            grid.connectivities[dims.E2VDim] == -1
        ):
            pytest.xfail("Stencil does not support missing neighbors.")

        vn_ie = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_vt_ie = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_primal_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        tangent_orientation = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_w_v = random_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        z_v_grad_w = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            vn_ie=vn_ie,
            inv_dual_edge_length=inv_dual_edge_length,
            w=w,
            z_vt_ie=z_vt_ie,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_w_v=z_w_v,
            z_v_grad_w=z_v_grad_w,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
