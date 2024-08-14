# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02c import face_val_ppm_stencil_02c
from icon4py.model.common.dimension import CellDim, KDim
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
        p_cc = random_field(grid, CellDim, KDim)
        p_face = random_field(grid, CellDim, KDim)
        return dict(
            p_cc=p_cc,
            p_face=p_face,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
