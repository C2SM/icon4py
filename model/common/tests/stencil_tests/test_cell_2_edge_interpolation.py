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

from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
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
        e2c = grid.connectivities[E2CDim]
        coeff_ = np.expand_dims(coeff, axis=-1)
        out_field = np.sum(in_field[e2c] * coeff_, axis=1)

        return dict(
            out_field=out_field,
        )

    @pytest.fixture
    def input_data(self, grid):
        in_field = random_field(grid, CellDim, KDim, dtype=wpfloat)
        coeff = random_field(grid, EdgeDim, E2CDim, dtype=wpfloat)
        out_field = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            in_field=in_field,
            coeff=coeff,
            out_field=out_field,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
