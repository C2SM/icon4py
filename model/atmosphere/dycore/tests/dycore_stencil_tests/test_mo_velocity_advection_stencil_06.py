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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_06 import (
    mo_velocity_advection_stencil_06,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_velocity_advection_stencil_06_numpy(wgtfacq_e: np.array, vn: np.array) -> np.array:
    vn_k_minus_1 = np.roll(vn, shift=1, axis=1)
    vn_k_minus_2 = np.roll(vn, shift=2, axis=1)
    vn_k_minus_3 = np.roll(vn, shift=3, axis=1)
    wgtfacq_e_k_minus_1 = np.roll(wgtfacq_e, shift=1, axis=1)
    wgtfacq_e_k_minus_2 = np.roll(wgtfacq_e, shift=2, axis=1)
    wgtfacq_e_k_minus_3 = np.roll(wgtfacq_e, shift=3, axis=1)
    vn_ie = np.zeros_like(vn)
    vn_ie[:, -1] = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )[:, -1]
    return vn_ie


class TestMoVelocityAdvectionStencil06(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_06
    OUTPUTS = ("vn_ie",)

    @staticmethod
    def reference(mesh, wgtfacq_e: np.array, vn: np.array, **kwargs) -> dict:
        vn_ie = mo_velocity_advection_stencil_06_numpy(wgtfacq_e, vn)
        return dict(vn_ie=vn_ie)

    @pytest.fixture
    def input_data(self, grid):
        wgtfacq_e = random_field(grid, EdgeDim, KDim)
        vn = random_field(grid, EdgeDim, KDim)
        vn_ie = zero_field(grid, EdgeDim, KDim)

        return dict(
            wgtfacq_e=wgtfacq_e,
            vn=vn,
            vn_ie=vn_ie,
        )
