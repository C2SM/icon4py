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

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestAddAnalysisIncrementsToVn(StencilTest):
    PROGRAM = add_analysis_increments_to_vn
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(grid, vn_incr: np.array, vn: np.array, iau_wgt_dyn, **kwargs) -> dict:
        vn = vn + (iau_wgt_dyn * vn_incr)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        vn_incr = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        iau_wgt_dyn = wpfloat("5.0")

        return dict(
            vn_incr=vn_incr,
            vn=vn,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
