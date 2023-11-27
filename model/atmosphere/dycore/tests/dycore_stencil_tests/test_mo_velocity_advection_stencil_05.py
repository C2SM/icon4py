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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_05 import (
    mo_velocity_advection_stencil_05,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def mo_velocity_advection_stencil_05_numpy(vn: np.array, vt: np.array) -> tuple:
    vn_ie = vn
    z_vt_ie = vt
    z_kin_hor_e = 0.5 * ((vn * vn) + (vt * vt))
    return vn_ie, z_vt_ie, z_kin_hor_e


class TestMoVelocityAdvectionStencil05(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_05
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(grid, vn: np.array, vt: np.array, **kwargs) -> dict:
        vn_ie, z_vt_ie, z_kin_hor_e = mo_velocity_advection_stencil_05_numpy(vn, vt)
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        vn_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_vt_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
        )
