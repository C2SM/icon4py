# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_tendency import (
    compute_tendency,
)
from icon4py.model.common import dimension as dims


class TestComputeTendency(helpers.StencilTest):
    PROGRAM = compute_tendency
    OUTPUTS = ("opt_ddt_tracer_adv",)

    @staticmethod
    def reference(
        grid,
        p_tracer_now: np.array,
        p_tracer_new: np.array,
        p_dtime,
        **kwargs,
    ) -> dict:
        opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime

        return dict(opt_ddt_tracer_adv=opt_ddt_tracer_adv)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_tracer_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_tracer_new = helpers.random_field(grid, dims.CellDim, dims.KDim)
        opt_ddt_tracer_adv = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = np.float64(5.0)
        return dict(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            p_dtime=p_dtime,
            opt_ddt_tracer_adv=opt_ddt_tracer_adv,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
