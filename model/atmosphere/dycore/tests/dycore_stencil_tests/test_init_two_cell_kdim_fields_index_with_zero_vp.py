# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest as pytest

from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_index_with_zero_vp import (
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
        k1: gtx.int32,
        k2: gtx.int32,
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

        k = gtx.as_field((dims.KDim,), np.arange(0, _shape(grid, dims.KDim)[0], dtype=gtx.int32))
        k1 = 1
        k2 = gtx.int32(grid.num_levels)

        return dict(
            field_index_with_zero_1=field_index_with_zero_1,
            field_index_with_zero_2=field_index_with_zero_2,
            k=k,
            k1=k1,
            k2=k2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
