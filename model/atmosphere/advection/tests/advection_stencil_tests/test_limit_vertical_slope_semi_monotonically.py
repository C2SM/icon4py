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
from gt4py.next import as_field

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.limit_vertical_slope_semi_monotonically import (
    limit_vertical_slope_semi_monotonically,
)
from icon4py.model.common import dimension as dims


class TestLimitVerticalSlopeSemiMonotonically(helpers.StencilTest):
    PROGRAM = limit_vertical_slope_semi_monotonically
    OUTPUTS = (helpers.Output("z_slope", gtslice=(slice(None), slice(1, -1))),)

    @staticmethod
    def reference(
        grid, p_cc: np.array, z_slope: np.array, k: np.array, elev: gtx.int32, **kwargs
    ) -> dict:
        p_cc_min_last = np.minimum(p_cc[:, :-2], p_cc[:, 1:-1])
        p_cc_min = np.where(k[1:-1] == elev, p_cc_min_last, np.minimum(p_cc_min_last, p_cc[:, 2:]))
        slope_l = np.minimum(np.abs(z_slope[:, 1:-1]), 2.0 * (p_cc[:, 1:-1] - p_cc_min))
        slope = np.where(z_slope[:, 1:-1] >= 0.0, slope_l, -slope_l)
        return dict(z_slope=slope)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_slope = helpers.random_field(grid, dims.CellDim, dims.KDim)
        k = as_field(
            (dims.KDim,), np.arange(0, helpers._shape(grid, dims.KDim)[0], dtype=gtx.int32)
        )
        elev = k[-2].as_scalar()
        return dict(
            p_cc=p_cc,
            z_slope=z_slope,
            k=k,
            elev=elev,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels - 1),
        )
