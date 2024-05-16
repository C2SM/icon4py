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

from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


def init_cell_kdim_field_with_zero_vp_numpy(field_with_zero_vp: np.array) -> np.array:
    field_with_zero_vp = np.zeros_like(field_with_zero_vp)
    return field_with_zero_vp


class TestInitCellKdimFieldWithZeroVp(StencilTest):
    PROGRAM = init_cell_kdim_field_with_zero_vp
    OUTPUTS = ("field_with_zero_vp",)

    @staticmethod
    def reference(grid, field_with_zero_vp: np.array, **kwargs) -> dict:
        field_with_zero_vp = init_cell_kdim_field_with_zero_vp_numpy(
            field_with_zero_vp=field_with_zero_vp
        )
        return dict(field_with_zero_vp=field_with_zero_vp)

    @pytest.fixture
    def input_data(self, grid):
        field_with_zero_vp = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            field_with_zero_vp=field_with_zero_vp,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
