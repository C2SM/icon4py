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

from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_with_zero_wp import (
    init_two_cell_kdim_fields_with_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestInitTwoCellKdimFieldsWithZeroWp(StencilTest):
    PROGRAM = init_two_cell_kdim_fields_with_zero_wp
    OUTPUTS = ("cell_kdim_field_with_zero_wp_1", "cell_kdim_field_with_zero_wp_2")

    @staticmethod
    def reference(
        grid,
        cell_kdim_field_with_zero_wp_1: np.array,
        cell_kdim_field_with_zero_wp_2: np.array,
        **kwargs,
    ) -> dict:
        cell_kdim_field_with_zero_wp_1 = np.zeros_like(cell_kdim_field_with_zero_wp_1)
        cell_kdim_field_with_zero_wp_2 = np.zeros_like(cell_kdim_field_with_zero_wp_2)
        return dict(
            cell_kdim_field_with_zero_wp_1=cell_kdim_field_with_zero_wp_1,
            cell_kdim_field_with_zero_wp_2=cell_kdim_field_with_zero_wp_2,
        )

    @pytest.fixture
    def input_data(self, grid):
        cell_kdim_field_with_zero_wp_1 = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        cell_kdim_field_with_zero_wp_2 = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            cell_kdim_field_with_zero_wp_1=cell_kdim_field_with_zero_wp_1,
            cell_kdim_field_with_zero_wp_2=cell_kdim_field_with_zero_wp_2,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
