# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def calculate_horizontal_gradients_for_turbulence_numpy(grid, w, geofac_grg_x, geofac_grg_y):
    c2e2cO = grid.connectivities[dims.C2E2CODim]
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
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        dwdx = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        dwdy = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w=w,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
