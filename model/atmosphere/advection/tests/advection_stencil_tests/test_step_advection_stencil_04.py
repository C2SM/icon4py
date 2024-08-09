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

from icon4py.model.atmosphere.advection.step_advection_stencil_04 import step_advection_stencil_04
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestStepAdvectionStencil04(StencilTest):
    PROGRAM = step_advection_stencil_04
    OUTPUTS = ("opt_ddt_tracer_adv",)

    @staticmethod
    def reference(
        grid,
        p_tracer_now: np.array,
        p_tracer_new: np.array,
        p_dtime,
        **kwargs,
    ):
        opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime

        return dict(opt_ddt_tracer_adv=opt_ddt_tracer_adv)

    @pytest.fixture
    def input_data(self, grid):
        p_tracer_now = random_field(grid, dims.CellDim, dims.KDim)
        p_tracer_new = random_field(grid, dims.CellDim, dims.KDim)
        opt_ddt_tracer_adv = zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = np.float64(5.0)
        return dict(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            p_dtime=p_dtime,
            opt_ddt_tracer_adv=opt_ddt_tracer_adv,
        )
