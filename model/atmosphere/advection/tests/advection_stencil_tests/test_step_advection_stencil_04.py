# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.step_advection_stencil_04 import (
    step_advection_stencil_04,
)
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
