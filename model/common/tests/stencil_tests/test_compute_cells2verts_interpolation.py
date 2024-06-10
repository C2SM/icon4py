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

from icon4py.model.common.dimension import CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.interpolation.stencils.compute_cells2verts_interpolation import (
    compute_cells2verts_interpolation,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestComputeCells2VertsInterpolation(StencilTest):
    PROGRAM = compute_cells2verts_interpolation
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
        cell_in = random_field(grid, CellDim, KDim, dtype=wpfloat)
        c_int = random_field(grid, VertexDim, V2CDim, dtype=wpfloat)
        vert_out = zero_field(grid, VertexDim, KDim, dtype=wpfloat)

        return dict(
            cell_in=cell_in,
            c_int=c_int,
            vert_out=vert_out,
            horizontal_start=0,
            horizontal_end=int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
