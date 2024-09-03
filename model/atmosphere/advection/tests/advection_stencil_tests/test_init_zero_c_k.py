# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.advection.stencils.init_zero_c_k import init_zero_c_k
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
