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
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_64 import (
    mo_solve_nonhydro_stencil_64,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil64(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_64
    OUTPUTS = ("mass_flx_ic",)

    @staticmethod
    def reference(grid, mass_flx_ic: np.array, **kwargs) -> dict:
        mass_flx_ic = np.zeros_like(mass_flx_ic)
        return dict(mass_flx_ic=mass_flx_ic)

    @pytest.fixture
    def input_data(self, grid):
        mass_flx_ic = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            mass_flx_ic=mass_flx_ic,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
