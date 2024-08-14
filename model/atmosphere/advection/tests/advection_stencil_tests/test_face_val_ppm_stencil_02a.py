# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02a import face_val_ppm_stencil_02a
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import Output, StencilTest, random_field


outslice = (slice(None), slice(1, None))


class TestFaceValPpmStencil02a(StencilTest):
    PROGRAM = face_val_ppm_stencil_02a
    OUTPUTS = (Output("p_face", refslice=outslice, gtslice=outslice),)

    @staticmethod
    def reference(grid, p_cc: np.array, p_cellhgt_mc_now: np.array, **kwargs):
        p_face = p_cc.copy()
        p_face[:, 1:] = p_cc[:, 1:] * (
            1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])
        ) + (p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])) * (
            (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1]
        )
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_face = random_field(grid, CellDim, KDim)
        p_cc = random_field(grid, CellDim, KDim)
        p_cellhgt_mc_now = random_field(grid, CellDim, KDim)
        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            p_face=p_face,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=1,
            vertical_end=grid.num_levels,
        )
