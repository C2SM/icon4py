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

from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoMathGradientsGradGreenGaussCellDsl(StencilTest):
    PROGRAM = mo_math_gradients_grad_green_gauss_cell_dsl
    OUTPUTS = ("p_grad_1_u", "p_grad_1_v", "p_grad_2_u", "p_grad_2_v")

    @staticmethod
    def reference(
        grid,
        p_ccpr1: np.array,
        p_ccpr2: np.array,
        geofac_grg_x: np.array,
        geofac_grg_y: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        c2e2cO = grid.connectivities[dims.C2E2CODim]
        geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
        p_grad_1_u = np.sum(
            np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * p_ccpr1[c2e2cO], 0), axis=1
        )
        geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
        p_grad_1_v = np.sum(
            np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * p_ccpr1[c2e2cO], 0), axis=1
        )
        p_grad_2_u = np.sum(
            np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * p_ccpr2[c2e2cO], 0), axis=1
        )
        p_grad_2_v = np.sum(
            np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * p_ccpr2[c2e2cO], 0), axis=1
        )
        return dict(
            p_grad_1_u=p_grad_1_u,
            p_grad_1_v=p_grad_1_v,
            p_grad_2_u=p_grad_2_u,
            p_grad_2_v=p_grad_2_v,
        )

    @pytest.fixture
    def input_data(self, grid):
        p_ccpr1 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        p_ccpr2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        p_grad_1_u = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        p_grad_1_v = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        p_grad_2_u = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        p_grad_2_v = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            p_grad_1_u=p_grad_1_u,
            p_grad_1_v=p_grad_1_v,
            p_grad_2_u=p_grad_2_u,
            p_grad_2_v=p_grad_2_v,
            p_ccpr1=p_ccpr1,
            p_ccpr2=p_ccpr2,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
