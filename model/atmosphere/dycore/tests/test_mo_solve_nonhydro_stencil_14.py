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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_14 import (
    mo_solve_nonhydro_stencil_14,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import zero_field
from icon4py.model.common.test_utils.benchmark import StencilTest


class TestMoSolveNonhydroStencil14(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_14
    OUTPUTS = ("z_rho_e", "z_theta_v_e")

    @staticmethod
    def reference(mesh, z_rho_e: np.array, z_theta_v_e: np.array, **kwargs) -> dict:
        z_rho_e = np.zeros_like(z_rho_e)
        z_theta_v_e = np.zeros_like(z_theta_v_e)
        return dict(z_rho_e=z_rho_e, z_theta_v_e=z_theta_v_e)

    @pytest.fixture
    def input_data(self, mesh):
        z_rho_e = zero_field(mesh, EdgeDim, KDim)
        z_theta_v_e = zero_field(mesh, EdgeDim, KDim)

        return dict(
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
        )
