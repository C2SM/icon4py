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

from icon4py.model.atmosphere.advection.hflx_limiter_mo_stencil_04 import hflx_limiter_mo_stencil_04
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestHflxLimiterMoStencil04(StencilTest):
    PROGRAM = hflx_limiter_mo_stencil_04
    OUTPUTS = ("p_mflx_tracer_h",)

    @staticmethod
    def reference(
        grid,
        z_anti: np.ndarray,
        r_m: np.ndarray,
        r_p: np.ndarray,
        z_mflx_low: np.ndarray,
        **kwargs,
    ):
        r_frac = np.where(
            z_anti >= 0,
            np.minimum(
                r_m[grid.connectivities[E2CDim][:, 0]],
                r_p[grid.connectivities[E2CDim][:, 1]],
            ),
            np.minimum(
                r_m[grid.connectivities[E2CDim][:, 1]],
                r_p[grid.connectivities[E2CDim][:, 0]],
            ),
        )
        return dict(p_mflx_tracer_h=z_mflx_low + np.minimum(1.0, r_frac) * z_anti)

    @pytest.fixture
    def input_data(self, grid):
        z_anti = random_field(grid, EdgeDim, KDim, low=-2.0, high=2.0)
        r_m = random_field(grid, CellDim, KDim)
        r_p = random_field(grid, CellDim, KDim)
        z_mflx_low = random_field(grid, EdgeDim, KDim)
        p_mflx_tracer_h = zero_field(grid, EdgeDim, KDim)
        return dict(
            z_anti=z_anti,
            r_m=r_m,
            r_p=r_p,
            z_mflx_low=z_mflx_low,
            p_mflx_tracer_h=p_mflx_tracer_h,
        )
