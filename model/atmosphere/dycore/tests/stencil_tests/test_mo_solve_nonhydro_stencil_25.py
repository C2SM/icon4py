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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_25 import (
    mo_solve_nonhydro_stencil_25,
)
from icon4py.model.common.dimension import E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_solve_nonhydro_stencil_25_numpy(
    grid, geofac_grdiv: np.array, z_graddiv_vn: np.array
) -> np.array:
    e2c2eO = grid.connectivities[E2C2EODim]
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_graddiv2_vn = np.sum(
        np.where((e2c2eO != -1)[:, :, np.newaxis], z_graddiv_vn[e2c2eO] * geofac_grdiv, 0),
        axis=1,
    )
    return z_graddiv2_vn


class TestMoSolveNonhydroStencil25(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_25
    OUTPUTS = ("z_graddiv2_vn",)

    @staticmethod
    def reference(grid, geofac_grdiv: np.array, z_graddiv_vn: np.array, **kwargs) -> np.array:
        z_graddiv2_vn = mo_solve_nonhydro_stencil_25_numpy(grid, geofac_grdiv, z_graddiv_vn)
        return dict(z_graddiv2_vn=z_graddiv2_vn)

    @pytest.fixture
    def input_data(self, grid):
        z_graddiv_vn = random_field(grid, EdgeDim, KDim)
        geofac_grdiv = random_field(grid, EdgeDim, E2C2EODim)
        z_graddiv2_vn = zero_field(grid, EdgeDim, KDim)

        return dict(
            geofac_grdiv=geofac_grdiv,
            z_graddiv_vn=z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
