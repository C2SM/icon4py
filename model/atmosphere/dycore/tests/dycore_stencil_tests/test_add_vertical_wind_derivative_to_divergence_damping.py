# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.dycore.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestAddVerticalWindDerivativeToDivergenceDamping(StencilTest):
    PROGRAM = add_vertical_wind_derivative_to_divergence_damping
    OUTPUTS = ("z_graddiv_vn",)

    @staticmethod
    def reference(
        grid,
        hmask_dd3d: np.array,
        scalfac_dd3d: np.array,
        inv_dual_edge_length: np.array,
        z_dwdz_dd: np.array,
        z_graddiv_vn: np.array,
        **kwargs,
    ) -> dict:
        scalfac_dd3d = np.expand_dims(scalfac_dd3d, axis=0)
        hmask_dd3d = np.expand_dims(hmask_dd3d, axis=-1)
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        z_dwdz_dd_e2c = z_dwdz_dd[grid.connectivities[dims.E2CDim]]
        z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1] - z_dwdz_dd_e2c[:, 0]

        z_graddiv_vn = z_graddiv_vn + (
            hmask_dd3d * scalfac_dd3d * inv_dual_edge_length * z_dwdz_dd_weighted
        )
        return dict(z_graddiv_vn=z_graddiv_vn)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        hmask_dd3d = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        scalfac_dd3d = random_field(grid, dims.KDim, dtype=wpfloat)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_dwdz_dd = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_graddiv_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
