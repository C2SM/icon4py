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

from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestSolveTridiagonalMatrixForWBackSubstitution(StencilTest):
    PROGRAM = solve_tridiagonal_matrix_for_w_back_substitution
    OUTPUTS = ("w",)

    @staticmethod
    def reference(grid, z_q: np.array, w: np.array, **kwargs) -> dict:
        w_new = np.zeros_like(w)
        last_k_level = w.shape[1] - 1

        w_new[:, last_k_level] = w[:, last_k_level]
        for k in reversed(range(1, last_k_level)):
            w_new[:, k] = w[:, k] + w_new[:, k + 1] * z_q[:, k]
        w_new[:, 0] = w[:, 0]
        return dict(w=w_new)

    @pytest.fixture
    def input_data(self, grid):
        z_q = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        h_start = 0
        h_end = gtx.int32(grid.num_cells)
        v_start = 1
        v_end = gtx.int32(grid.num_levels)
        return dict(
            z_q=z_q,
            w=w,
            horizontal_start=h_start,
            horizontal_end=h_end,
            vertical_start=v_start,
            vertical_end=v_end,
        )
