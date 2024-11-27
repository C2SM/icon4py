# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as xp
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_vertical_parabola_limiter_condition import (
    compute_vertical_parabola_limiter_condition,
)
from icon4py.model.common import dimension as dims


class TestComputeVerticalParabolaLimiterCondition(helpers.StencilTest):
    PROGRAM = compute_vertical_parabola_limiter_condition
    OUTPUTS = ("l_limit",)

    @staticmethod
    def reference(grid, p_face: xp.array, p_cc: xp.array, **kwargs) -> dict:
        z_delta = p_face[:, :-1] - p_face[:, 1:]
        z_a6i = 6.0 * (p_cc - 0.5 * (p_face[:, :-1] + p_face[:, 1:]))
        l_limit = xp.where(xp.abs(z_delta) < -1 * z_a6i, 1, 0)
        return dict(l_limit=l_limit)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_face = helpers.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        l_limit = helpers.zero_field(grid, dims.CellDim, dims.KDim, dtype=gtx.int32)
        return dict(
            p_face=p_face,
            p_cc=p_cc,
            l_limit=l_limit,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
