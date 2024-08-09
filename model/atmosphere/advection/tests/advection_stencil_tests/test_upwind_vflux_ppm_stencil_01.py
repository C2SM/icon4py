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

from icon4py.model.atmosphere.advection.upwind_vflux_ppm_stencil_01 import (
    upwind_vflux_ppm_stencil_01,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestUpwindVfluxPpmStencil01(StencilTest):
    PROGRAM = upwind_vflux_ppm_stencil_01
    OUTPUTS = ("z_delta_q", "z_a1")

    @staticmethod
    def reference(
        grid, z_face_up: np.ndarray, z_face_low: np.ndarray, p_cc: np.ndarray, **kwargs
    ) -> tuple[np.ndarray]:
        z_delta_q = 0.5 * (z_face_up - z_face_low)
        z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)
        return dict(z_delta_q=z_delta_q, z_a1=z_a1)

    @pytest.fixture
    def input_data(self, grid):
        z_face_up = random_field(grid, dims.CellDim, dims.KDim)
        z_face_low = random_field(grid, dims.CellDim, dims.KDim)
        p_cc = random_field(grid, dims.CellDim, dims.KDim)
        z_delta_q = zero_field(grid, dims.CellDim, dims.KDim)
        z_a1 = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            z_face_up=z_face_up, z_face_low=z_face_low, p_cc=p_cc, z_delta_q=z_delta_q, z_a1=z_a1
        )
