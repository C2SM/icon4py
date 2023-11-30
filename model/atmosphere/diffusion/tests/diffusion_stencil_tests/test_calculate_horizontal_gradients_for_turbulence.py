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

from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def calculate_horizontal_gradients_for_turbulence_numpy(grid, w, geofac_grg_x, geofac_grg_y):
    c2e2cO = grid.connectivities[C2E2CODim]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    dwdx = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * w[c2e2cO], 0.0), axis=1)

    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    dwdy = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * w[c2e2cO], 0.0), axis=1)
    return dwdx, dwdy


class TestCalculateHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = calculate_horizontal_gradients_for_turbulence
    OUTPUTS = ("dwdx", "dwdy")

    @staticmethod
    def reference(
        grid, w: np.array, geofac_grg_x: np.array, geofac_grg_y: np.array, **kwargs
    ) -> dict:
        dwdx, dwdy = calculate_horizontal_gradients_for_turbulence_numpy(
            grid, w, geofac_grg_x, geofac_grg_y
        )
        return dict(dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, grid):
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        geofac_grg_x = random_field(grid, CellDim, C2E2CODim, dtype=wpfloat)
        geofac_grg_y = random_field(grid, CellDim, C2E2CODim, dtype=wpfloat)
        dwdx = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        dwdy = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            w=w,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
