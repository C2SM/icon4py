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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_60 import (
    mo_solve_nonhydro_stencil_60,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil60(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_60
    OUTPUTS = ("exner_dyn_incr",)

    @staticmethod
    def reference(
        grid,
        exner: np.array,
        ddt_exner_phy: np.array,
        exner_dyn_incr: np.array,
        ndyn_substeps_var: float,
        dtime: float,
        **kwargs,
    ) -> np.array:
        exner_dyn_incr = exner - (exner_dyn_incr + ndyn_substeps_var * dtime * ddt_exner_phy)
        return dict(exner_dyn_incr=exner_dyn_incr)

    @pytest.fixture
    def input_data(self, grid):
        ndyn_substeps_var, dtime = 10.0, 12.0
        exner = random_field(grid, CellDim, KDim)
        ddt_exner_phy = random_field(grid, CellDim, KDim)
        exner_dyn_incr = random_field(grid, CellDim, KDim)

        return dict(
            exner=exner,
            ddt_exner_phy=ddt_exner_phy,
            exner_dyn_incr=exner_dyn_incr,
            ndyn_substeps_var=ndyn_substeps_var,
            dtime=dtime,
        )
