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

from icon4py.model.atmosphere.dycore.compute_first_vertical_derivative import (
    compute_first_vertical_derivative,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


class TestComputeFirstVerticalDerivative(StencilTest):
    PROGRAM = compute_first_vertical_derivative
    OUTPUTS = ("z_dexner_dz_c_1",)

    @staticmethod
    def reference(grid, z_exner_ic: np.array, inv_ddqz_z_full: np.array, **kwargs) -> dict:
        z_dexner_dz_c_1 = (z_exner_ic[:, :-1] - z_exner_ic[:, 1:]) * inv_ddqz_z_full
        return dict(z_dexner_dz_c_1=z_dexner_dz_c_1)

    @pytest.fixture
    def input_data(self, grid):
        z_exner_ic = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=vpfloat)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_dexner_dz_c_1 = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            z_exner_ic=z_exner_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
