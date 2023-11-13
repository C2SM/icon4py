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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_12 import (
    mo_solve_nonhydro_stencil_12,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


class TestMoSolveNonhydroStencil12(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_12
    OUTPUTS = ("z_dexner_dz_c_2",)

    @staticmethod
    def reference(
        grid,
        z_theta_v_pr_ic: np.array,
        d2dexdz2_fac1_mc: np.array,
        d2dexdz2_fac2_mc: np.array,
        z_rth_pr_2: np.array,
        **kwargs,
    ) -> np.array:
        z_theta_v_pr_ic_offset_1 = z_theta_v_pr_ic[:, 1:]
        z_dexner_dz_c_2 = -0.5 * (
            (z_theta_v_pr_ic[:, :-1] - z_theta_v_pr_ic_offset_1) * d2dexdz2_fac1_mc
            + z_rth_pr_2 * d2dexdz2_fac2_mc
        )
        return dict(z_dexner_dz_c_2=z_dexner_dz_c_2)

    @pytest.fixture
    def input_data(self, grid):
        z_theta_v_pr_ic = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=vpfloat)
        d2dexdz2_fac1_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rth_pr_2 = random_field(grid, CellDim, KDim, dtype=vpfloat)
        d2dexdz2_fac2_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)

        z_dexner_dz_c_2 = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            z_rth_pr_2=z_rth_pr_2,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
