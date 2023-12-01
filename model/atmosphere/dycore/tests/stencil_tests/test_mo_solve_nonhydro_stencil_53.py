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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_53 import (
    mo_solve_nonhydro_stencil_53,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil53(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_53
    OUTPUTS = ("w",)

    @staticmethod
    def reference(grid, z_q: np.array, w: np.array, **kwargs) -> np.array:
        w_new = np.zeros_like(w)
        last_k_level = w.shape[1] - 1

        w_new[:, last_k_level] = w[:, last_k_level]
        for k in reversed(range(1, last_k_level)):
            w_new[:, k] = w[:, k] + w_new[:, k + 1] * z_q[:, k]
        w_new[:, 0] = w[:, 0]
        return dict(w=w_new)

    @pytest.fixture
    def input_data(self, grid):
        z_q = random_field(grid, CellDim, KDim, dtype=vpfloat)
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        h_start = int32(0)
        h_end = int32(grid.num_cells)
        v_start = int32(1)
        v_end = int32(grid.num_levels)
        return dict(
            z_q=z_q,
            w=w,
            horizontal_start=h_start,
            horizontal_end=h_end,
            vertical_start=v_start,
            vertical_end=v_end,
        )
