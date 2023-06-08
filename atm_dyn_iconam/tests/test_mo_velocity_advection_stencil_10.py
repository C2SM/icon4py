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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_10 import (
    mo_velocity_advection_stencil_10,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field, zero_field


class TestMoVelocityAdvectionStencil10(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_10
    OUTPUTS = ("w_concorr_c",)

    @staticmethod
    def reference(
        mesh, wgtfac_c: np.array, z_w_concorr_mc: np.array, **kwargs
    ) -> np.array:
        z_w_concorr_mc_k_minus_1 = np.roll(z_w_concorr_mc, shift=1, axis=1)
        w_concorr_c = (
            wgtfac_c * z_w_concorr_mc + (1.0 - wgtfac_c) * z_w_concorr_mc_k_minus_1
        )

        return dict(w_concorr_c=w_concorr_c)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfac_c = random_field(mesh, CellDim, KDim)
        z_w_concorr_mc = random_field(mesh, CellDim, KDim)

        w_concorr_c = zero_field(mesh, CellDim, KDim)

        return dict(
            z_w_concorr_mc=z_w_concorr_mc,
            wgtfac_c=wgtfac_c,
            w_concorr_c=w_concorr_c,
        )
