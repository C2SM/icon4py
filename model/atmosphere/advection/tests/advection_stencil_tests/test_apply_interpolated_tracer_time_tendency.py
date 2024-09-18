# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.stencils.apply_interpolated_tracer_time_tendency import (
    apply_interpolated_tracer_time_tendency,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestApplyInterpolatedTracerTimeTendency(StencilTest):
    PROGRAM = apply_interpolated_tracer_time_tendency
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
        p_tracer_now = random_field(grid, dims.CellDim, dims.KDim)
        p_grf_tend_tracer = random_field(grid, dims.CellDim, dims.KDim)
        p_tracer_new = random_field(grid, dims.CellDim, dims.KDim)
        p_dtime = np.float64(5.0)
        return dict(
            p_tracer_now=p_tracer_now,
            p_grf_tend_tracer=p_grf_tend_tracer,
            p_dtime=p_dtime,
            p_tracer_new=p_tracer_new,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )