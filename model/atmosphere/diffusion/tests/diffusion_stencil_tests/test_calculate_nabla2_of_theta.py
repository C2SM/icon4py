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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    calculate_nabla2_of_theta,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)


def calculate_nabla2_of_theta_numpy(grid, z_nabla2_e: np.array, geofac_div: np.array) -> np.array:
    c2e = grid.connectivities[C2EDim]
    geofac_div = geofac_div.reshape(c2e.shape)
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


class TestCalculateNabla2OfTheta(StencilTest):
    PROGRAM = calculate_nabla2_of_theta
    OUTPUTS = ("z_temp",)

    @staticmethod
    def reference(grid, z_nabla2_e: np.array, geofac_div: np.array, **kwargs) -> dict:
        z_temp = calculate_nabla2_of_theta_numpy(grid, z_nabla2_e, geofac_div)
        return dict(z_temp=z_temp)

    @pytest.fixture
    def input_data(self, grid):
        z_nabla2_e = random_field(grid, EdgeDim, KDim)
        geofac_div = random_field(grid, CellDim, C2EDim)
        geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)

        z_temp = zero_field(grid, CellDim, KDim)

        return dict(z_nabla2_e=z_nabla2_e, geofac_div=geofac_div_new, z_temp=z_temp)
