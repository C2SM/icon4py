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

from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_to_zero_wp import (
    set_two_cell_kdim_fields_to_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil46(StencilTest):
    PROGRAM = set_two_cell_kdim_fields_to_zero_wp
    OUTPUTS = ("w_nnew", "z_contr_w_fl_l")

    @staticmethod
    def reference(grid, w_nnew: np.array, z_contr_w_fl_l: np.array, **kwargs) -> dict:
        w_nnew = np.zeros_like(w_nnew)
        z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
        return dict(w_nnew=w_nnew, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        z_contr_w_fl_l = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        w_nnew = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            w_nnew=w_nnew,
            z_contr_w_fl_l=z_contr_w_fl_l,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
