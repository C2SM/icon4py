# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


class TestMoMathGradientsGradGreenGaussCellDsl(StencilTest):
    PROGRAM = mo_math_gradients_grad_green_gauss_cell_dsl
    OUTPUTS = ("p_grad_1_u", "p_grad_1_v", "p_grad_2_u", "p_grad_2_v")
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_ccpr1: np.ndarray,
        p_ccpr2: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs,
    ) -> dict:
        c2e2cO = connectivities[dims.C2E2CODim]
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
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
