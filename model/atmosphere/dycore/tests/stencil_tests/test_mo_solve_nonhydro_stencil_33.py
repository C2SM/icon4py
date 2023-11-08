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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_33 import (
    mo_solve_nonhydro_stencil_33,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field


class TestMoSolveNonhydroStencil33(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_33
    OUTPUTS = ("vn_traj", "mass_flx_me")

    @staticmethod
    def reference(grid, vn_traj: np.array, mass_flx_me: np.array, **kwargs) -> dict:
        vn_traj = np.zeros_like(vn_traj)
        mass_flx_me = np.zeros_like(mass_flx_me)
        return dict(vn_traj=vn_traj, mass_flx_me=mass_flx_me)

    @pytest.fixture
    def input_data(self, grid):
        vn_traj = zero_field(grid, EdgeDim, KDim)
        mass_flx_me = zero_field(grid, EdgeDim, KDim)

        return dict(
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
