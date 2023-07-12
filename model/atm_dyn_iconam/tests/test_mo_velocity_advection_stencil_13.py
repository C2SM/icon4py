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

from icon4py.model.atm_dyn_iconam.mo_velocity_advection_stencil_13 import (
    mo_velocity_advection_stencil_13,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field
from .test_utils.stencil_test import StencilTest


class TestMoVelocityAdvectionStencil13(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_13
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(
        mesh, w_concorr_c: np.array, z_w_con_c: np.array, **kwargs
    ) -> np.array:
        z_w_con_c = z_w_con_c - w_concorr_c
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, mesh):
        z_w_con_c = random_field(mesh, CellDim, KDim)
        w_concorr_c = random_field(mesh, CellDim, KDim)

        return dict(
            w_concorr_c=w_concorr_c,
            z_w_con_c=z_w_con_c,
        )
