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

from icon4py.model.atmosphere.advection.v_limit_prbl_sm_stencil_01 import v_limit_prbl_sm_stencil_01
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestVLimitPrblSmStencil01(StencilTest):
    PROGRAM = v_limit_prbl_sm_stencil_01
    OUTPUTS = ("l_limit",)

    @staticmethod
    def reference(grid, p_face: np.array, p_cc: np.array, **kwargs):
        z_delta = p_face[:, :-1] - p_face[:, 1:]
        z_a6i = 6.0 * (p_cc - 0.5 * (p_face[:, :-1] + p_face[:, 1:]))
        l_limit = np.where(np.abs(z_delta) < -1 * z_a6i, 1, 0)
        return dict(l_limit=l_limit)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, dims.CellDim, dims.KDim)
        p_face = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        l_limit = zero_field(grid, dims.CellDim, dims.KDim, dtype=int32)
        return dict(p_face=p_face, p_cc=p_cc, l_limit=l_limit)
