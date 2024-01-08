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

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02c import face_val_ppm_stencil_02c
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestFaceValPpmStencil02c(StencilTest):
    PROGRAM = face_val_ppm_stencil_02c
    OUTPUTS = (("p_face", (slice(None), slice(1, None))),)

    @staticmethod
    def reference(grid, p_cc: np.array, **kwargs):
        p_face = p_cc.copy()
        p_face[:, 1:] = p_cc[:, :-1]
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, CellDim, KDim)
        p_face = random_field(grid, CellDim, KDim)
        return dict(p_cc=p_cc, p_face=p_face)
