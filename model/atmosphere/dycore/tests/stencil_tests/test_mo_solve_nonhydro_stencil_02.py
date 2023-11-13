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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_02 import (
    mo_solve_nonhydro_stencil_02,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil02(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_02
    OUTPUTS = ("z_exner_ex_pr", "exner_pr")

    @staticmethod
    def reference(
        grid,
        exner: np.array,
        exner_ref_mc: np.array,
        exner_pr: np.array,
        exner_exfac: np.array,
        **kwargs,
    ) -> dict:
        z_exner_ex_pr = (1 + exner_exfac) * (exner - exner_ref_mc) - exner_exfac * exner_pr
        exner_pr = exner - exner_ref_mc
        return dict(z_exner_ex_pr=z_exner_ex_pr, exner_pr=exner_pr)

    @pytest.fixture
    def input_data(self, grid):
        exner = random_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_ref_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        exner_pr = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_exfac = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_exner_ex_pr = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            exner_exfac=exner_exfac,
            exner=exner,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
            z_exner_ex_pr=z_exner_ex_pr,
        )
