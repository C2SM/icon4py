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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_34 import (
    mo_solve_nonhydro_stencil_34,
)
from icon4py.model.common.dimension import EdgeDim, KDim

from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestMoSolveNonhydroStencil34(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_34
    OUTPUTS = ("vn_traj", "mass_flx_me")

    @staticmethod
    def reference(
        mesh,
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
    def input_data(self, mesh):
        mass_fl_e = random_field(mesh, EdgeDim, KDim)
        mass_flx_me = random_field(mesh, EdgeDim, KDim)
        z_vn_avg = random_field(mesh, EdgeDim, KDim)
        vn_traj = random_field(mesh, EdgeDim, KDim)
        r_nsubsteps = 9.0

        return dict(
            z_vn_avg=z_vn_avg,
            mass_fl_e=mass_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            r_nsubsteps=r_nsubsteps,
        )
