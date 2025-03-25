# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest as pytest

from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_index_with_zero_vp import (
    init_two_cell_kdim_fields_index_with_zero_vp,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestInitTwoCellKdimFieldsIndexWithZeroVp(StencilTest):
    PROGRAM = init_two_cell_kdim_fields_index_with_zero_vp
    OUTPUTS = ("field_index_with_zero_1", "field_index_with_zero_2")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        field_index_with_zero_1: np.ndarray,
        field_index_with_zero_2: np.ndarray,
        k: np.ndarray,
        k1: gtx.int32,
        k2: gtx.int32,
        **kwargs: Any,
    ) -> dict:
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
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        field_index_with_zero_1 = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        field_index_with_zero_2 = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=vpfloat
        )
        k = data_alloc.index_field(grid, dims.KDim)

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
