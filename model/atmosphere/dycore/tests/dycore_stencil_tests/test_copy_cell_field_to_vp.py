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

from icon4py.model.atmosphere.dycore.copy_cell_field_to_vp import copy_cell_field_to_vp
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def copy_cell_field_to_vp_numpy(w: np.array) -> np.array:
    z_w_con_c = w
    return z_w_con_c


class TestMoVelocityAdvectionStencil11(StencilTest):
    PROGRAM = copy_cell_field_to_vp
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(grid, w: np.array, **kwargs) -> dict:
        z_w_con_c = copy_cell_field_to_vp_numpy(w)
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, grid):
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        z_w_con_c = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        return dict(
            w=w,
            z_w_con_c=z_w_con_c,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
