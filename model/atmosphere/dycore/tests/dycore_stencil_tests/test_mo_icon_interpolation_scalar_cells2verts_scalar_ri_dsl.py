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

from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common.dimension import CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
    grid, p_cell_in: np.array, c_intp: np.array
) -> np.array:
    v2c = grid.connectivities[V2CDim]
    c_intp = np.expand_dims(c_intp, axis=-1)
    p_vert_out = np.sum(np.where((v2c != -1)[:, :, np.newaxis], p_cell_in[v2c] * c_intp, 0), axis=1)
    return p_vert_out


class TestMoIconInterpolationScalarCells2vertsScalarRiDsl(StencilTest):
    PROGRAM = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl
    OUTPUTS = ("p_vert_out",)

    @staticmethod
    def reference(grid, p_cell_in: np.array, c_intp: np.array, **kwargs) -> dict:
        p_vert_out = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            grid, p_cell_in, c_intp
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
            horizontal_start=0,
            horizontal_end=int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
