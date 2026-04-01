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

from icon4py.model.atmosphere.dycore.stencils.compute_vn_on_lateral_boundary import (
    compute_vn_on_lateral_boundary,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests


class TestComputeVnOnLateralBoundary(stencil_tests.StencilTest):
    PROGRAM = compute_vn_on_lateral_boundary
    OUTPUTS = ("vn_new",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        grf_tend_vn: np.ndarray,
        vn_now: np.ndarray,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn_new = vn_now + dtime * grf_tend_vn
        return dict(vn_new=vn_new)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        grf_tend_vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        vn_now = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        vn_new = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("6.0")

        return dict(
            grf_tend_vn=grf_tend_vn,
            vn_now=vn_now,
            vn_new=vn_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
