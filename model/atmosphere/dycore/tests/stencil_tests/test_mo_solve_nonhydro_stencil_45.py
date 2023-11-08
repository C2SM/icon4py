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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_45 import (
    mo_solve_nonhydro_stencil_45,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field


class TestMoSolveNonhydroStencil45(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_45
    OUTPUTS = ("z_alpha",)

    @staticmethod
    def reference(grid, z_alpha: np.array, **kwargs) -> dict:
        z_alpha = np.zeros_like(z_alpha)
        return dict(z_alpha=z_alpha)

    @pytest.fixture
    def input_data(self, grid):
        z_alpha = zero_field(grid, CellDim, KDim)

        return dict(
            z_alpha=z_alpha,
        )
