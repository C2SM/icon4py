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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_29 import (
    mo_solve_nonhydro_stencil_29,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil29(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_29
    OUTPUTS = ("vn_new",)

    @staticmethod
    def reference(grid, grf_tend_vn: np.array, vn_now: np.array, dtime, **kwargs) -> dict:
        vn_new = vn_now + dtime * grf_tend_vn
        return dict(vn_new=vn_new)

    @pytest.fixture
    def input_data(self, grid):
        grf_tend_vn = random_field(grid, EdgeDim, KDim)
        vn_now = random_field(grid, EdgeDim, KDim)
        vn_new = zero_field(grid, EdgeDim, KDim)
        dtime = 6.0

        return dict(
            grf_tend_vn=grf_tend_vn,
            vn_now=vn_now,
            vn_new=vn_new,
            dtime=dtime,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
