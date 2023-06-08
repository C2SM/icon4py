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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_11_lower import (
    mo_solve_nonhydro_stencil_11_lower,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field


class TestMoSolveNonhydroStencil11Lower(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_11_lower
    OUTPUTS = ("z_theta_v_pr_ic",)

    @staticmethod
    def reference(mesh, **kwargs) -> np.array:
        z_theta_v_pr_ic = 0
        return dict(z_theta_v_pr_ic=z_theta_v_pr_ic)

    @pytest.fixture
    def input_data(self, mesh):
        z_theta_v_pr_ic = random_field(mesh, CellDim, KDim)
        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
        )
