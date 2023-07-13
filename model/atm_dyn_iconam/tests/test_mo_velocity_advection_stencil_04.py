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

from icon4py.model.atm_dyn_iconam.mo_velocity_advection_stencil_04 import (
    mo_velocity_advection_stencil_04,
)
from icon4py.model.common.dimension import EdgeDim, KDim

from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestMoVelocityAdvectionStencil04(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_04
    OUTPUTS = ("z_w_concorr_me",)

    @staticmethod
    def reference(
        mesh,
        vn: np.array,
        ddxn_z_full: np.array,
        ddxt_z_full: np.array,
        vt: np.array,
        **kwargs,
    ) -> np.array:
        z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
        return dict(z_w_concorr_me=z_w_concorr_me)

    @pytest.fixture
    def input_data(self, mesh):
        vn = random_field(mesh, EdgeDim, KDim)
        ddxn_z_full = random_field(mesh, EdgeDim, KDim)
        ddxt_z_full = random_field(mesh, EdgeDim, KDim)
        vt = random_field(mesh, EdgeDim, KDim)
        z_w_concorr_me = zero_field(mesh, EdgeDim, KDim)

        return dict(
            vn=vn,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            vt=vt,
            z_w_concorr_me=z_w_concorr_me,
        )
