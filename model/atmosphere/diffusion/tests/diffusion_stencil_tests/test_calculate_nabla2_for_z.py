# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as xp
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    calculate_nabla2_for_z,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def calculate_nabla2_for_z_numpy(
    grid,
    kh_smag_e: xp.array,
    inv_dual_edge_length: xp.array,
    theta_v: xp.array,
    z_nabla2_e: xp.array,
    **kwargs,
) -> xp.array:
    z_nabla2_e_cp = xp.asarray(z_nabla2_e.copy())
    inv_dual_edge_length = xp.expand_dims(xp.asarray(inv_dual_edge_length), axis=-1)

    theta_v_e2c = xp.asarray(theta_v)[xp.asarray(grid.connectivities[dims.E2CDim])]
    theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

    z_nabla2_e = xp.asarray(kh_smag_e) * inv_dual_edge_length * theta_v_weighted

    # restriction of execution domain
    z_nabla2_e[0 : kwargs["horizontal_start"], :] = z_nabla2_e_cp[0 : kwargs["horizontal_start"], :]
    z_nabla2_e[kwargs["horizontal_end"] :, :] = z_nabla2_e_cp[kwargs["horizontal_end"] :, :]
    return z_nabla2_e


class TestCalculateNabla2ForZ(StencilTest):
    PROGRAM = calculate_nabla2_for_z
    OUTPUTS = ("z_nabla2_e",)

    @staticmethod
    def reference(
        grid,
        kh_smag_e: xp.array,
        inv_dual_edge_length: xp.array,
        theta_v: xp.array,
        z_nabla2_e: xp.array,
        **kwargs,
    ) -> dict:
        z_nabla2_e = calculate_nabla2_for_z_numpy(
            grid, kh_smag_e, inv_dual_edge_length, theta_v, z_nabla2_e, **kwargs
        )
        return dict(z_nabla2_e=z_nabla2_e)

    @pytest.fixture
    def input_data(self, grid):
        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = (
            grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
            if hasattr(grid, "start_index")
            else 0
        )

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v,
            z_nabla2_e=z_nabla2_e,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
