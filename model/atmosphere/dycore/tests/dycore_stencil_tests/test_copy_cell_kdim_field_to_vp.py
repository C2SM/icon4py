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

from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def copy_cell_kdim_field_to_vp_numpy(field: np.array) -> np.array:
    field_copy = field
    return field_copy


class TestCopyCellKdimFieldToVp(StencilTest):
    PROGRAM = copy_cell_kdim_field_to_vp
    OUTPUTS = ("field_copy",)

    @staticmethod
    def reference(grid, field: np.array, **kwargs) -> dict:
        field_copy = copy_cell_kdim_field_to_vp_numpy(field)
        return dict(field_copy=field_copy)

    @pytest.fixture
    def input_data(self, grid):
        field = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        field_copy = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        return dict(
            field=field,
            field_copy=field_copy,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
