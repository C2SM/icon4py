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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_11 import (
    mo_velocity_advection_stencil_11,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_velocity_advection_stencil_11_numpy(w: np.array) -> np.array:
    z_w_con_c = w
    return z_w_con_c


class TestMoVelocityAdvectionStencil11(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_11
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(grid, w: np.array, **kwargs) -> dict:
        z_w_con_c = mo_velocity_advection_stencil_11_numpy(w)
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, grid):
        w = random_field(grid, CellDim, KDim)
        z_w_con_c = zero_field(grid, CellDim, KDim)
        return dict(w=w, z_w_con_c=z_w_con_c)
