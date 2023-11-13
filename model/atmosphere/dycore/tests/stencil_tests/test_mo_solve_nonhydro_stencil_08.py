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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_08 import (
    mo_solve_nonhydro_stencil_08,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil08(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_08
    OUTPUTS = ("rho_ic", "z_rth_pr_1", "z_rth_pr_2")

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        rho = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_ref_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        theta_v = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_ref_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        rho_ic = zero_field(grid, CellDim, KDim, dtype=wpfloat)
        z_rth_pr_1 = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rth_pr_2 = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            rho=rho,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            rho_ic=rho_ic,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
        )

    @staticmethod
    def reference(
        grid,
        wgtfac_c: np.array,
        rho: np.array,
        rho_ref_mc: np.array,
        theta_v: np.array,
        theta_ref_mc: np.array,
        **kwargs,
    ) -> tuple[np.array, np.array, np.array]:
        rho_offset_1 = np.roll(rho, shift=1, axis=1)
        rho_ic = wgtfac_c * rho + (1.0 - wgtfac_c) * rho_offset_1
        rho_ic[:, 0] = 0
        z_rth_pr_1 = rho - rho_ref_mc
        z_rth_pr_1[:, 0] = 0
        z_rth_pr_2 = theta_v - theta_ref_mc
        z_rth_pr_2[:, 0] = 0

        return dict(rho_ic=rho_ic, z_rth_pr_1=z_rth_pr_1, z_rth_pr_2=z_rth_pr_2)
