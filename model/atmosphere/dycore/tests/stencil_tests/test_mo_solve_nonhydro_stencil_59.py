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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_59 import (
    mo_solve_nonhydro_stencil_59,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def mo_solve_nonhydro_stencil_59_numpy(grid, exner: np.array) -> np.array:
    exner_dyn_incr = exner
    return exner_dyn_incr


class TestMoSolveNonhydroStencil59(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_59
    OUTPUTS = ("exner_dyn_incr",)

    @staticmethod
    def reference(grid, exner: np.array, **kwargs) -> dict:
        exner_dyn_incr = mo_solve_nonhydro_stencil_59_numpy(grid, exner)
        return dict(exner_dyn_incr=exner_dyn_incr)

    @pytest.fixture
    def input_data(self, grid):
        exner = random_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_dyn_incr = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            exner=exner,
            exner_dyn_incr=exner_dyn_incr,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
