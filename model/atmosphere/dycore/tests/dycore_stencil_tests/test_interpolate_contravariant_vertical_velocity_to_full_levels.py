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

from icon4py.model.atmosphere.dycore.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


def interpolate_contravariant_vertical_velocity_to_full_levels_numpy(grid, z_w_con_c: np.array):
    z_w_con_c_full = 0.5 * (z_w_con_c[:, :-1] + z_w_con_c[:, 1:])
    return z_w_con_c_full


class TestInterpolateContravariantVerticalVelocityToFullLevels(StencilTest):
    PROGRAM = interpolate_contravariant_vertical_velocity_to_full_levels
    OUTPUTS = ("z_w_con_c_full",)

    @staticmethod
    def reference(grid, z_w_con_c: np.array, **kwargs) -> dict:
        z_w_con_c_full = interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
            grid, z_w_con_c
        )
        return dict(z_w_con_c_full=z_w_con_c_full)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat
        )

        z_w_con_c_full = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            z_w_con_c=z_w_con_c,
            z_w_con_c_full=z_w_con_c_full,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
