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

from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import random_field, zero_field, StencilTest



class TestMoMathGradientsGradGreenGaussCellDsl(StencilTest):
    PROGRAM = mo_math_gradients_grad_green_gauss_cell_dsl
    OUTPUTS = ("p_grad_1_u", "p_grad_1_v", "p_grad_2_u", "p_grad_2_v")

    @staticmethod
    def reference(
        mesh,
        p_ccpr1: np.array,
        p_ccpr2: np.array,
        geofac_grg_x: np.array,
        geofac_grg_y: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
        p_grad_1_u = np.sum(geofac_grg_x * p_ccpr1[mesh.c2e2cO], axis=1)

        geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
        p_grad_1_v = np.sum(geofac_grg_y * p_ccpr1[mesh.c2e2cO], axis=1)

        p_grad_2_u = np.sum(geofac_grg_x * p_ccpr2[mesh.c2e2cO], axis=1)

        p_grad_2_v = np.sum(geofac_grg_y * p_ccpr2[mesh.c2e2cO], axis=1)
        return dict(
            p_grad_1_u=p_grad_1_u,
            p_grad_1_v=p_grad_1_v,
            p_grad_2_u=p_grad_2_u,
            p_grad_2_v=p_grad_2_v,
        )

    @pytest.fixture
    def input_data(self, mesh):
        p_ccpr1 = random_field(mesh, CellDim, KDim)
        p_ccpr2 = random_field(mesh, CellDim, KDim)
        geofac_grg_x = random_field(mesh, CellDim, C2E2CODim)
        geofac_grg_y = random_field(mesh, CellDim, C2E2CODim)
        p_grad_1_u = zero_field(mesh, CellDim, KDim)
        p_grad_1_v = zero_field(mesh, CellDim, KDim)
        p_grad_2_u = zero_field(mesh, CellDim, KDim)
        p_grad_2_v = zero_field(mesh, CellDim, KDim)

        return dict(
            p_grad_1_u=p_grad_1_u,
            p_grad_1_v=p_grad_1_v,
            p_grad_2_u=p_grad_2_u,
            p_grad_2_v=p_grad_2_v,
            p_ccpr1=p_ccpr1,
            p_ccpr2=p_ccpr2,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
        )
