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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_15 import (
    mo_velocity_advection_stencil_15,
)
from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestMoVelocityAdvectionStencil15(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_15
    OUTPUTS = ("z_w_con_c_full",)

    @staticmethod
    def reference(mesh, z_w_con_c: np.array, **kwargs):
        z_w_con_c_full = 0.5 * (z_w_con_c[:, :-1] + z_w_con_c[:, 1:])
        return dict(z_w_con_c_full=z_w_con_c_full)

    @pytest.fixture
    def input_data(self, mesh):
        z_w_con_c = random_field(mesh, CellDim, KDim, extend={KDim: 1})

        z_w_con_c_full = zero_field(mesh, CellDim, KDim)

        return dict(
            z_w_con_c=z_w_con_c,
            z_w_con_c_full=z_w_con_c_full,
        )
