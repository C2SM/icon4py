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
import pytest as pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_to_zero_vp import (
    set_two_cell_kdim_fields_to_zero_vp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


class TestSetTwoCellKdimFieldsToZeroVp(StencilTest):
    PROGRAM = set_two_cell_kdim_fields_to_zero_vp
    OUTPUTS = ("cell_kdim_field_to_zero_vp_1", "cell_kdim_field_to_zero_vp_2")

    @staticmethod
    def reference(
        grid,
        cell_kdim_field_to_zero_vp_1: np.array,
        cell_kdim_field_to_zero_vp_2: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        cell_kdim_field_to_zero_vp_1 = np.zeros_like(cell_kdim_field_to_zero_vp_1)
        cell_kdim_field_to_zero_vp_2 = np.zeros_like(cell_kdim_field_to_zero_vp_2)
        return dict(
            cell_kdim_field_to_zero_vp_1=cell_kdim_field_to_zero_vp_1,
            cell_kdim_field_to_zero_vp_2=cell_kdim_field_to_zero_vp_2,
        )

    @pytest.fixture
    def input_data(self, grid):
        cell_kdim_field_to_zero_vp_1 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        cell_kdim_field_to_zero_vp_2 = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            cell_kdim_field_to_zero_vp_1=cell_kdim_field_to_zero_vp_1,
            cell_kdim_field_to_zero_vp_2=cell_kdim_field_to_zero_vp_2,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
