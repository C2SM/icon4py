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

from icon4py.model.common import dimension as dims
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestCell2EdgeInterpolation(StencilTest):
    PROGRAM = cell_2_edge_interpolation
    OUTPUTS = ("out_field",)

    @staticmethod
    def reference(grid, in_field: np.array, coeff: np.array, **kwargs) -> dict:
        e2c = grid.connectivities[dims.E2CDim]
        coeff_ = np.expand_dims(coeff, axis=-1)
        out_field = np.sum(in_field[e2c] * coeff_, axis=1)

        return dict(
            out_field=out_field,
        )

    @pytest.fixture
    def input_data(self, grid):
        in_field = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        coeff = random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=wpfloat)
        out_field = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            in_field=in_field,
            coeff=coeff,
            out_field=out_field,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
