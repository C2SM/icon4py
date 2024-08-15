# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.face_val_ppm_stencil_02b import (
    face_val_ppm_stencil_02b,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestFaceValPpmStencil02b(StencilTest):
    PROGRAM = face_val_ppm_stencil_02b
    OUTPUTS = ("p_face",)

    @staticmethod
    def reference(grid, p_cc: np.array, **kwargs):
        p_face = p_cc.copy()
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, CellDim, KDim)
        p_face = random_field(grid, CellDim, KDim)
        return dict(p_cc=p_cc, p_face=p_face)
