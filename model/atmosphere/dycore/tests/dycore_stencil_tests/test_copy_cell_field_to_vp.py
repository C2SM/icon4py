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

from icon4py.model.atmosphere.dycore.copy_cell_field_to_vp import (
    copy_cell_field_to_vp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def copy_cell_field_to_vp_numpy(field: np.array) -> np.array:
    field_copy = field
    return field_copy


class TestMoVelocityAdvectionStencil11(StencilTest):
    PROGRAM = copy_cell_field_to_vp
    OUTPUTS = ("field_copy",)

    @staticmethod
    def reference(grid, field: np.array, **kwargs) -> dict:
        field_copy = copy_cell_field_to_vp_numpy(field)
        return dict(field_copy=field_copy)

    @pytest.fixture
    def input_data(self, grid):
        field = random_field(grid, CellDim, KDim, dtype=wpfloat)
        field_copy = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        return dict(
            field=field,
            field_copy=field_copy,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
