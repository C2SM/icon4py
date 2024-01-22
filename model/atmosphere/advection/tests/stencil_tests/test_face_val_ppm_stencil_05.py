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

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_05 import face_val_ppm_stencil_05
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import Output, StencilTest, random_field, zero_field


class TestFaceValPpmStencil05(StencilTest):
    PROGRAM = face_val_ppm_stencil_05
    OUTPUTS = (Output("p_face", gtslice=(slice(None), slice(2, None))),)

    @staticmethod
    def reference(grid, p_cc: np.array, p_cellhgt_mc_now: np.array, z_slope: np.array, **kwargs):
        p_cellhgt_mc_now_k_minus_1 = p_cellhgt_mc_now[:, 1:-2]
        p_cellhgt_mc_now_k_minus_2 = p_cellhgt_mc_now[:, 0:-3]
        p_cellhgt_mc_now_k_plus_1 = p_cellhgt_mc_now[:, 3:]
        p_cellhgt_mc_now = p_cellhgt_mc_now[:, 2:-1]

        p_cc_k_minus_1 = p_cc[:, 1:-1]
        p_cc = p_cc[:, 2:]
        z_slope_k_minus_1 = z_slope[:, 1:-1]
        z_slope = z_slope[:, 2:]

        zgeo1 = p_cellhgt_mc_now_k_minus_1 / (p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now)
        zgeo2 = 1.0 / (
            p_cellhgt_mc_now_k_minus_2
            + p_cellhgt_mc_now_k_minus_1
            + p_cellhgt_mc_now
            + p_cellhgt_mc_now_k_plus_1
        )
        zgeo3 = (p_cellhgt_mc_now_k_minus_2 + p_cellhgt_mc_now_k_minus_1) / (
            2.0 * p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now
        )
        zgeo4 = (p_cellhgt_mc_now_k_plus_1 + p_cellhgt_mc_now) / (
            2 * p_cellhgt_mc_now + p_cellhgt_mc_now_k_minus_1
        )

        p_face = (
            p_cc_k_minus_1
            + zgeo1 * (p_cc - p_cc_k_minus_1)
            + zgeo2
            * (
                (2 * p_cellhgt_mc_now * zgeo1) * (zgeo3 - zgeo4) * (p_cc - p_cc_k_minus_1)
                - zgeo3 * p_cellhgt_mc_now_k_minus_1 * z_slope
                + zgeo4 * p_cellhgt_mc_now * z_slope_k_minus_1
            )
        )
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, CellDim, KDim)
        p_cellhgt_mc_now = random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_slope = random_field(grid, CellDim, KDim)
        p_face = zero_field(grid, CellDim, KDim)

        return dict(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, z_slope=z_slope, p_face=p_face)
