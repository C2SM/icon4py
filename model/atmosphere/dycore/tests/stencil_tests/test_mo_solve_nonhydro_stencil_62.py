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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_62 import (
    mo_solve_nonhydro_stencil_62,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil62(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_62
    OUTPUTS = ("w_new",)

    @staticmethod
    def reference(grid, w_now: np.array, grf_tend_w: np.array, dtime: float, **kwargs) -> np.array:
        w_new = w_now + dtime * grf_tend_w
        return dict(w_new=w_new)

    @pytest.fixture
    def input_data(self, grid):
        dtime = 10.0
        w_now = random_field(grid, CellDim, KDim)
        grf_tend_w = random_field(grid, CellDim, KDim)
        w_new = zero_field(grid, CellDim, KDim)

        return dict(
            w_now=w_now,
            grf_tend_w=grf_tend_w,
            w_new=w_new,
            dtime=dtime,
        )
