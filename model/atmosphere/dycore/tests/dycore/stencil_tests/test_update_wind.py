# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.update_wind import update_wind
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestUpdateWind(stencil_tests.StencilTest):
    PROGRAM = update_wind
    OUTPUTS = ("w_new",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        w_now: np.ndarray,
        grf_tend_w: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        w_new = w_now + dtime * grf_tend_w
        return dict(w_new=w_new)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = wpfloat("10.0")
        w_now = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        grf_tend_w = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        w_new = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            w_now=w_now,
            grf_tend_w=grf_tend_w,
            w_new=w_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
