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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeHorizontalGradientOfExnerPressureForNonflatCoordinates(StencilTest):
    PROGRAM = compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        grid,
        inv_dual_edge_length: np.array,
        z_exner_ex_pr: np.array,
        ddxn_z_full: np.array,
        c_lin_e: np.array,
        z_dexner_dz_c_1: np.array,
        **kwargs,
    ) -> dict:
        e2c = grid.connectivities[dims.E2CDim]
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
        c_lin_e = np.expand_dims(c_lin_e, axis=-1)

        z_exner_ex_pr_e2c = z_exner_ex_pr[e2c]
        z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

        z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted - ddxn_z_full * np.sum(
            c_lin_e * z_dexner_dz_c_1[e2c], axis=1
        )
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[dims.E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_exner_ex_pr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddxn_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        c_lin_e = random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=wpfloat)
        z_dexner_dz_c_1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_gradh_exner = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
