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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    calculate_nabla2_for_w,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KHalfDim
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field


def calculate_nabla2_for_w_numpy(grid, w: np.array, geofac_n2s: np.array):
    c2e2cO = grid.connectivities[C2E2CODim]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    z_nabla2_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], w[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return z_nabla2_c


class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = calculate_nabla2_for_w
    OUTPUTS = ("z_nabla2_c",)

    @staticmethod
    def reference(grid, w: np.array, geofac_n2s: np.array, **kwargs) -> dict:
        z_nabla2_c = calculate_nabla2_for_w_numpy(grid, w, geofac_n2s)
        return dict(z_nabla2_c=z_nabla2_c)

    @pytest.fixture
    def input_data(self, grid):
        w = constant_field(grid, 1.0, CellDim, KHalfDim)
        geofac_n2s = constant_field(grid, 2.0, CellDim, C2E2CODim)
        z_nabla2_c = zero_field(grid, CellDim, KHalfDim)

        return dict(
            w=w,
            geofac_n2s=geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
