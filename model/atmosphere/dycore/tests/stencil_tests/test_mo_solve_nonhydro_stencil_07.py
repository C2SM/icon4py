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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_07 import (
    mo_solve_nonhydro_stencil_07,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil07(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_07
    OUTPUTS = ("z_rth_pr_1", "z_rth_pr_2")

    @staticmethod
    def reference(
        grid,
        rho: np.array,
        rho_ref_mc: np.array,
        theta_v: np.array,
        theta_ref_mc: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        z_rth_pr_1 = rho - rho_ref_mc
        z_rth_pr_2 = theta_v - theta_ref_mc
        return dict(z_rth_pr_1=z_rth_pr_1, z_rth_pr_2=z_rth_pr_2)

    @pytest.fixture
    def input_data(self, grid):
        rho = random_field(grid, CellDim, KDim)
        rho_ref_mc = random_field(grid, CellDim, KDim)
        theta_v = random_field(grid, CellDim, KDim)
        theta_ref_mc = random_field(grid, CellDim, KDim)
        z_rth_pr_1 = zero_field(grid, CellDim, KDim)
        z_rth_pr_2 = zero_field(grid, CellDim, KDim)

        return dict(
            rho=rho,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
        )
