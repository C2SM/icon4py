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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_03 import (
    mo_velocity_advection_stencil_03,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_velocity_advection_stencil_03_numpy(
    grid, wgtfac_e: np.array, vt: np.array, **kwargs
) -> np.array:
    vt_k_minus_1 = np.roll(vt, shift=1, axis=1)
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_k_minus_1
    z_vt_ie[:, 0] = 0
    return z_vt_ie


class TestMoVelocityAdvectionStencil03(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_03
    OUTPUTS = ("z_vt_ie",)

    @staticmethod
    def reference(grid, wgtfac_e: np.array, vt: np.array, **kwargs) -> dict:
        z_vt_ie = mo_velocity_advection_stencil_03_numpy(grid, wgtfac_e, vt)
        return dict(
            z_vt_ie=z_vt_ie[int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)]
        )

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_e = random_field(grid, EdgeDim, KDim)
        vt = random_field(grid, EdgeDim, KDim)

        z_vt_ie = zero_field(grid, EdgeDim, KDim)

        return dict(
            wgtfac_e=wgtfac_e,
            vt=vt,
            z_vt_ie=z_vt_ie[int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)],
            horizontal_start=int32(1),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
