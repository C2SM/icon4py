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

from icon4py.model.common.dimension import C2E2C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.interpolation.stencils.rbf_vec_interpol_edge2cell import (
    rbf_vec_interpol_edge2cell,
)
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoIntpRbfVecInterpolCell(StencilTest):
    PROGRAM = rbf_vec_interpol_edge2cell
    OUTPUTS = ("p_u_out", "p_v_out")

    @staticmethod
    def reference(
        grid, p_e_in: np.array, ptr_coeff_1: np.array, ptr_coeff_2: np.array, **kwargs
    ) -> tuple[np.array]:
        c2e2c2e = grid.connectivities[C2E2C2EDim]
        ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
        ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
        p_u_out = np.sum(p_e_in[c2e2c2e] * ptr_coeff_1, axis=1)
        p_v_out = np.sum(p_e_in[c2e2c2e] * ptr_coeff_2, axis=1)

        return dict(p_v_out=p_v_out, p_u_out=p_u_out)

    @pytest.fixture
    def input_data(self, grid):
        p_e_in = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        ptr_coeff_1 = random_field(grid, CellDim, C2E2C2EDim, dtype=wpfloat)
        ptr_coeff_2 = random_field(grid, CellDim, C2E2C2EDim, dtype=wpfloat)
        p_v_out = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        p_u_out = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            p_e_in=p_e_in,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            p_v_out=p_v_out,
            p_u_out=p_u_out,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
