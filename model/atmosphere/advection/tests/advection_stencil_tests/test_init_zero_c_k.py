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

import pytest

from icon4py.model.atmosphere.advection.init_zero_c_k import init_zero_c_k
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestInitZeroCK(StencilTest):
    PROGRAM = init_zero_c_k
    OUTPUTS = ("field",)

    @staticmethod
    def reference(grid, **kwargs):
        return dict(field=zero_field(grid, dims.CellDim, dims.KDim).asnumpy())

    @pytest.fixture
    def input_data(self, grid):
        field = random_field(grid, dims.CellDim, dims.KDim)
        return dict(field=field)
