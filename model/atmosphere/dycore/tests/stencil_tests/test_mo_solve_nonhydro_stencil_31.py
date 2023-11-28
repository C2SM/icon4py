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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_31 import (
    mo_solve_nonhydro_stencil_31,
)
from icon4py.model.common.dimension import E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil31(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_31
    OUTPUTS = ("z_vn_avg",)

    @staticmethod
    def reference(grid, e_flx_avg: np.array, vn: np.array, **kwargs) -> np.array:
        e2c2eO = grid.connectivities[E2C2EODim]
        geofac_grdiv = np.expand_dims(e_flx_avg, axis=-1)
        z_vn_avg = np.sum(
            np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * geofac_grdiv, 0), axis=1
        )
        return dict(z_vn_avg=z_vn_avg)

    @pytest.fixture
    def input_data(self, grid):
        e_flx_avg = random_field(grid, EdgeDim, E2C2EODim, dtype=wpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_vn_avg = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            z_vn_avg=z_vn_avg,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
