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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    calculate_nabla2_for_z,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def calculate_nabla2_for_z_numpy(
    grid,
    kh_smag_e: np.array,
    inv_dual_edge_length: np.array,
    theta_v: np.array,
) -> np.array:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    theta_v_e2c = theta_v[grid.connectivities[dims.E2CDim]]
    theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

    z_nabla2_e = kh_smag_e * inv_dual_edge_length * theta_v_weighted
    return z_nabla2_e


class TestCalculateNabla2ForZ(StencilTest):
    PROGRAM = calculate_nabla2_for_z
    OUTPUTS = ("z_nabla2_e",)

    @staticmethod
    def reference(
        grid,
        kh_smag_e: np.array,
        inv_dual_edge_length: np.array,
        theta_v: np.array,
        **kwargs,
    ) -> dict:
        z_nabla2_e = calculate_nabla2_for_z_numpy(grid, kh_smag_e, inv_dual_edge_length, theta_v)
        return dict(z_nabla2_e=z_nabla2_e)

    @pytest.fixture
    def input_data(self, grid):
        if isinstance(grid, IconGrid) and grid.limited_area:
            pytest.xfail(
                "Execution domain needs to be restricted or boundary taken into account in stencil."
            )

        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v,
            z_nabla2_e=z_nabla2_e,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
