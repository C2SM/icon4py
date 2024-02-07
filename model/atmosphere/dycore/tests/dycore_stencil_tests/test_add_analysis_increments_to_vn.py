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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def add_analysis_increments_to_vn_numpy(
    grid, vn_incr: np.array, vn: np.array, iau_wgt_dyn
) -> np.array:
    vn = vn + (iau_wgt_dyn * vn_incr)
    return vn


class TestMoSolveNonhydroStencil28(StencilTest):
    PROGRAM = add_analysis_increments_to_vn
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(grid, vn_incr: np.array, vn: np.array, iau_wgt_dyn, **kwargs) -> dict:
        vn = add_analysis_increments_to_vn_numpy(grid, vn_incr, vn, iau_wgt_dyn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        vn_incr = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        iau_wgt_dyn = wpfloat("5.0")

        return dict(
            vn_incr=vn_incr,
            vn=vn,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
