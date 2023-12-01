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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_03 import (
    mo_solve_nonhydro_stencil_03,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


class TestMoSolveNonhydroStencil03(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_03
    OUTPUTS = ("z_exner_ex_pr",)

    @staticmethod
    def reference(grid, z_exner_ex_pr: np.array, **kwargs) -> dict:
        z_exner_ex_pr = np.zeros_like(z_exner_ex_pr)
        return dict(z_exner_ex_pr=z_exner_ex_pr)

    @pytest.fixture
    def input_data(self, grid):
        z_exner_ex_pr = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            z_exner_ex_pr=z_exner_ex_pr,
        )
