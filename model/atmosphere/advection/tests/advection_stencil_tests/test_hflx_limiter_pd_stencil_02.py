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

from icon4py.model.atmosphere.advection.hflx_limiter_pd_stencil_02 import hflx_limiter_pd_stencil_02
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestHflxLimiterPdStencil02(StencilTest):
    PROGRAM = hflx_limiter_pd_stencil_02
    OUTPUTS = ("p_mflx_tracer_h",)

    @staticmethod
    def reference(grid, r_m, p_mflx_tracer_h, **kwargs):
        e2c = grid.connectivities[E2CDim]
        r_m_e2c = r_m[e2c]
        p_mflx_tracer_h_out = np.where(
            p_mflx_tracer_h >= 0,
            p_mflx_tracer_h * r_m_e2c[:, 0],
            p_mflx_tracer_h * r_m_e2c[:, 1],
        )
        return dict(p_mflx_tracer_h=p_mflx_tracer_h_out)

    @pytest.fixture(params=[("no_match", 4), ("everywhere_match", 7), ("partly_match", 4)])
    def input_data(self, request, grid):
        r_m = random_field(grid, CellDim, KDim)
        p_mflx_tracer_h_in = random_field(grid, EdgeDim, KDim)

        return dict(
            r_m=r_m,
            p_mflx_tracer_h=p_mflx_tracer_h_in,
        )
