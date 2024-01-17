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

from icon4py.model.atmosphere.dycore.set_two_edge_kdim_fields_to_zero_wp import (
    set_two_edge_kdim_fields_to_zero_wp,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil14(StencilTest):
    PROGRAM = set_two_edge_kdim_fields_to_zero_wp
    OUTPUTS = ("z_rho_e", "z_theta_v_e")

    @staticmethod
    def reference(grid, z_rho_e: np.array, z_theta_v_e: np.array, **kwargs) -> dict:
        z_rho_e = np.zeros_like(z_rho_e)
        z_theta_v_e = np.zeros_like(z_theta_v_e)
        return dict(z_rho_e=z_rho_e, z_theta_v_e=z_theta_v_e)

    @pytest.fixture
    def input_data(self, grid):
        z_rho_e = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_theta_v_e = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
