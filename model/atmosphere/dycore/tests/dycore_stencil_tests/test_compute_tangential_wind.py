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

from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import compute_tangential_wind
from icon4py.model.common.dimension import E2C2EDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_tangential_wind_numpy(grid, vn: np.array, rbf_vec_coeff_e: np.array) -> np.array:
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    e2c2e = grid.connectivities[E2C2EDim]
    vt = np.sum(np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1)
    return vt


class TestComputeTangentialWind(StencilTest):
    PROGRAM = compute_tangential_wind
    OUTPUTS = ("vt",)

    @staticmethod
    def reference(grid, vn: np.array, rbf_vec_coeff_e: np.array, **kwargs) -> dict:
        vt = compute_tangential_wind_numpy(grid, vn, rbf_vec_coeff_e)
        return dict(vt=vt)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        rbf_vec_coeff_e = random_field(grid, EdgeDim, E2C2EDim, dtype=wpfloat)
        vt = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            vt=vt,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
