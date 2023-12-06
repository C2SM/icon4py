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

from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common.dimension import CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoIconInterpolationScalarCells2vertsScalarRiDsl(StencilTest):
    PROGRAM = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl
    OUTPUTS = ("p_vert_out",)

    @staticmethod
    def reference(grid, p_cell_in: np.array, c_intp: np.array, **kwargs) -> dict:
        v2c = grid.connectivities[V2CDim]
        c_intp = np.expand_dims(c_intp, axis=-1)
        p_vert_out = np.sum(
            np.where((v2c != -1)[:, :, np.newaxis], p_cell_in[v2c] * c_intp, 0), axis=1
        )
        return dict(
            p_vert_out=p_vert_out,
        )

    @pytest.fixture
    def input_data(self, grid):
        p_cell_in = random_field(grid, CellDim, KDim, dtype=wpfloat)
        c_intp = random_field(grid, VertexDim, V2CDim, dtype=wpfloat)
        p_vert_out = zero_field(grid, VertexDim, KDim, dtype=vpfloat)

        return dict(
            p_cell_in=p_cell_in,
            c_intp=c_intp,
            p_vert_out=p_vert_out,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_vertices),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
