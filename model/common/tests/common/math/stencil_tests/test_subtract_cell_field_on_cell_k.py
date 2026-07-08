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
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils.generic_math_operations import subtract_cell_field_on_cell_k
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestSubtractCellFieldOnCellK(StencilTest):
    PROGRAM = subtract_cell_field_on_cell_k
    OUTPUTS = ("difference",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        minuend: np.ndarray,
        subtrahend_cell: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(difference=minuend - subtrahend_cell[:, np.newaxis])

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        minuend = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        )
        subtrahend_cell = data_alloc.random_field(
            grid, dims.CellDim, low=0.0, high=3.0e3, dtype=wpfloat
        )
        difference = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            minuend=minuend,
            subtrahend_cell=subtrahend_cell,
            difference=difference,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
