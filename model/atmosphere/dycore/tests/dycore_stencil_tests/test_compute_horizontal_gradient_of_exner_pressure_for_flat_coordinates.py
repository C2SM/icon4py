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

from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeHorizontalGradientOfExnerPressureForFlatCoordinates(StencilTest):
    PROGRAM = compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(grid, inv_dual_edge_length: np.array, z_exner_ex_pr: np.array, **kwargs) -> dict:
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        z_exner_ex_pr_e2c = z_exner_ex_pr[grid.connectivities[dims.E2CDim]]
        z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

        z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_exner_ex_pr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_gradh_exner = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
