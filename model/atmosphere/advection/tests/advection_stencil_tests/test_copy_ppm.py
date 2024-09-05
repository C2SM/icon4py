# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.copy_ppm import (
    copy_ppm,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestCopyPpm(StencilTest):
    PROGRAM = copy_ppm
    OUTPUTS = ("p_face",)

    @staticmethod
    def reference(grid, p_cc: np.array, **kwargs):
        p_face = p_cc.copy()
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, dims.CellDim, dims.KDim)
        p_face = random_field(grid, dims.CellDim, dims.KDim)
        return dict(p_cc=p_cc, p_face=p_face)
