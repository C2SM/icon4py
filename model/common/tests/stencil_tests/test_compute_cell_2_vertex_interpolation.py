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

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as test_helpers
import icon4py.model.common.type_alias as types
from icon4py.model.common.dimension import CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    compute_cell_2_vertex_interpolation,
)


class TestComputeCells2VertsInterpolation(test_helpers.StencilTest):
    PROGRAM = compute_cell_2_vertex_interpolation
    OUTPUTS = ("vert_out",)

    @staticmethod
    def reference(grid, cell_in: np.array, c_int: np.array, **kwargs) -> dict:
        v2c = grid.connectivities[V2CDim]
        c_int = np.expand_dims(c_int, axis=-1)
        out_field = np.sum(cell_in[v2c] * c_int, axis=1)

        return dict(
            vert_out=out_field,
        )

    @pytest.fixture
    def input_data(self, grid):
        cell_in = test_helpers.random_field(grid, CellDim, KDim, dtype=types.wpfloat)
        c_int = test_helpers.random_field(grid, VertexDim, V2CDim, dtype=types.wpfloat)
        vert_out = test_helpers.zero_field(grid, VertexDim, KDim, dtype=types.wpfloat)

        return dict(
            cell_in=cell_in,
            c_int=c_int,
            vert_out=vert_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )