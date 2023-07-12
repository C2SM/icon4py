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

from icon4py.model.atm_dyn_iconam.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.stencil_test import StencilTest


class TestCalculateHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = calculate_horizontal_gradients_for_turbulence
    OUTPUTS = ("dwdx", "dwdy")

    @staticmethod
    def reference(
        mesh, w: np.array, geofac_grg_x: np.array, geofac_grg_y: np.array, **kwargs
    ) -> tuple[np.array]:
        geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
        dwdx = np.sum(geofac_grg_x * w[mesh.c2e2cO], axis=1)

        geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
        dwdy = np.sum(geofac_grg_y * w[mesh.c2e2cO], axis=1)
        return dict(dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, mesh):
        w = random_field(mesh, CellDim, KDim)
        geofac_grg_x = random_field(mesh, CellDim, C2E2CODim)
        geofac_grg_y = random_field(mesh, CellDim, C2E2CODim)
        dwdx = zero_field(mesh, CellDim, KDim)
        dwdy = zero_field(mesh, CellDim, KDim)

        return dict(
            w=w,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            dwdx=dwdx,
            dwdy=dwdy,
        )
