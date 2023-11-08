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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_67 import (
    mo_solve_nonhydro_stencil_67,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil67(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_67
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        grid,
        rho: np.array,
        exner: np.array,
        rd_o_cvd: float,
        rd_o_p0ref: float,
        **kwargs,
    ) -> dict:
        theta_v = np.copy(exner)
        exner = np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * theta_v))
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid):
        rd_o_cvd = 10.0
        rd_o_p0ref = 20.0
        rho = random_field(grid, CellDim, KDim, low=1, high=2)
        theta_v = random_field(grid, CellDim, KDim, low=1, high=2)
        exner = random_field(grid, CellDim, KDim, low=1, high=2)

        return dict(
            rho=rho,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            rd_o_p0ref=rd_o_p0ref,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
