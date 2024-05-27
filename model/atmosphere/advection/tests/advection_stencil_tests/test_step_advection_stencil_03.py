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

from icon4py.model.atmosphere.advection.step_advection_stencil_03 import step_advection_stencil_03
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestStepAdvectionStencil03(StencilTest):
    PROGRAM = step_advection_stencil_03
    OUTPUTS = ("p_tracer_new",)

    @staticmethod
    def reference(
        grid,
        p_tracer_now: np.array,
        p_grf_tend_tracer: np.array,
        p_dtime,
        **kwargs,
    ):
        p_tracer_new = p_tracer_now + p_dtime * p_grf_tend_tracer
        p_tracer_new = np.where(p_tracer_new < 0.0, 0.0, p_tracer_new)

        return dict(p_tracer_new=p_tracer_new)

    @pytest.fixture
    def input_data(self, grid):
        p_tracer_now = random_field(grid, CellDim, KDim)
        p_grf_tend_tracer = random_field(grid, CellDim, KDim)
        p_tracer_new = random_field(grid, CellDim, KDim)
        p_dtime = np.float64(5.0)
        return dict(
            p_tracer_now=p_tracer_now,
            p_grf_tend_tracer=p_grf_tend_tracer,
            p_dtime=p_dtime,
            p_tracer_new=p_tracer_new,
        )
