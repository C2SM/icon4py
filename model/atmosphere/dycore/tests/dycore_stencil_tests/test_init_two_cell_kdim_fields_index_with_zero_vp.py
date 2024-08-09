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
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_index_with_zero_vp import (
    init_two_cell_kdim_fields_index_with_zero_vp,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, _shape, random_field
from icon4py.model.common.type_alias import vpfloat


class TestInitTwoCellKdimFieldsIndexWithZeroVp(StencilTest):
    PROGRAM = init_two_cell_kdim_fields_index_with_zero_vp
    OUTPUTS = ("field_index_with_zero_1", "field_index_with_zero_2")

    @staticmethod
    def reference(
        grid,
        field_index_with_zero_1: np.array,
        field_index_with_zero_2: np.array,
        k: np.array,
        k1: int32,
        k2: int32,
        **kwargs,
    ) -> tuple[np.array]:
        field_index_with_zero_1 = np.where(
            k == k1, np.zeros_like(field_index_with_zero_1), field_index_with_zero_1
        )
        field_index_with_zero_2 = np.where(
            k == k2, np.zeros_like(field_index_with_zero_2), field_index_with_zero_2
        )
        return dict(
            field_index_with_zero_1=field_index_with_zero_1,
            field_index_with_zero_2=field_index_with_zero_2,
        )

    @pytest.fixture
    def input_data(self, grid):
        field_index_with_zero_1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        field_index_with_zero_2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        k = as_field((dims.KDim,), np.arange(0, _shape(grid, dims.KDim)[0], dtype=int32))
        k1 = 1
        k2 = int32(grid.num_levels)

        return dict(
            field_index_with_zero_1=field_index_with_zero_1,
            field_index_with_zero_2=field_index_with_zero_2,
            k=k,
            k1=k1,
            k2=k2,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
