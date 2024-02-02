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

from icon4py.model.atmosphere.dycore.accumulate_prep_adv_fields import accumulate_prep_adv_fields
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil34(StencilTest):
    PROGRAM = accumulate_prep_adv_fields
    OUTPUTS = ("vn_traj", "mass_flx_me")

    @staticmethod
    def reference(
        grid,
        z_vn_avg: np.array,
        mass_fl_e: np.array,
        vn_traj: np.array,
        mass_flx_me: np.array,
        r_nsubsteps,
        **kwargs,
    ) -> dict:
        vn_traj = vn_traj + r_nsubsteps * z_vn_avg
        mass_flx_me = mass_flx_me + r_nsubsteps * mass_fl_e
        return dict(vn_traj=vn_traj, mass_flx_me=mass_flx_me)

    @pytest.fixture
    def input_data(self, grid):
        mass_fl_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        mass_flx_me = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_vn_avg = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vn_traj = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        r_nsubsteps = wpfloat("9.0")

        return dict(
            z_vn_avg=z_vn_avg,
            mass_fl_e=mass_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
