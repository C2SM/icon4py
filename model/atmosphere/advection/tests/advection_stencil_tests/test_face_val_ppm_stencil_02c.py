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

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02c import face_val_ppm_stencil_02c
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestFaceValPpmStencil02c(StencilTest):
    PROGRAM = face_val_ppm_stencil_02c
    OUTPUTS = ("p_face",)

    @staticmethod
    def reference(
        grid,
        p_cc: np.array,
        p_face: np.array,
        horizontal_start: int32,
        horizontal_end: int32,
        vertical_start: int32,
        vertical_end: int32,
    ):
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        p_face = p_face.copy()
        p_face[subset] = np.roll(p_cc, shift=1, axis=1)[subset]
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, dims.CellDim, dims.KDim)
        p_face = random_field(grid, dims.CellDim, dims.KDim)
        return dict(
            p_cc=p_cc,
            p_face=p_face,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
