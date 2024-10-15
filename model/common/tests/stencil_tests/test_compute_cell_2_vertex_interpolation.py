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

import icon4py.model.common.type_alias as types
from icon4py.model.common import dimension as dims
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.test_utils import helpers


class TestComputeCells2VertsInterpolation(helpers.StencilTest):
    PROGRAM = compute_cell_2_vertex_interpolation
    OUTPUTS = ("vert_out",)

    @staticmethod
    def reference(grid, cell_in: np.array, c_int: np.array, **kwargs) -> dict:
        v2c = grid.connectivities[dims.V2CDim]
        c_int = np.expand_dims(c_int, axis=-1)
        out_field = np.sum(cell_in[v2c] * c_int, axis=1)

        return dict(
            vert_out=out_field,
        )

    @pytest.fixture
    def input_data(self, grid):
        cell_in = helpers.random_field(grid, dims.CellDim, dims.KDim, dtype=types.wpfloat)
        c_int = helpers.random_field(grid, dims.VertexDim, dims.V2CDim, dtype=types.wpfloat)
        vert_out = helpers.zero_field(grid, dims.VertexDim, dims.KDim, dtype=types.wpfloat)

        return dict(
            cell_in=cell_in,
            c_int=c_int,
            vert_out=vert_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
