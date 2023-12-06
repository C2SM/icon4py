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

import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_11_lower import (
    mo_solve_nonhydro_stencil_11_lower,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


class TestMoSolveNonhydroStencil11Lower(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_11_lower
    OUTPUTS = ("z_theta_v_pr_ic",)

    @staticmethod
    def reference(grid, **kwargs) -> dict:
        z_theta_v_pr_ic = 0
        return dict(z_theta_v_pr_ic=z_theta_v_pr_ic)

    @pytest.fixture
    def input_data(self, grid):
        z_theta_v_pr_ic = random_field(grid, CellDim, KDim, dtype=vpfloat)
        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
