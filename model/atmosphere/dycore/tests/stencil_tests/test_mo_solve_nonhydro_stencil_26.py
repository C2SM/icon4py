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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_26 import (
    mo_solve_nonhydro_stencil_26,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


def mo_solve_nonhydro_stencil_26_numpy(
    grid, z_graddiv_vn: np.array, vn: np.array, scal_divdamp_o2: float
) -> np.array:
    vn = vn + (scal_divdamp_o2 * z_graddiv_vn)
    return vn


class TestMoSolveNonhydroStencil26(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_26
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        grid, z_graddiv_vn: np.array, vn: np.array, scal_divdamp_o2: float, **kwargs
    ) -> dict:
        vn = mo_solve_nonhydro_stencil_26_numpy(grid, z_graddiv_vn, vn, scal_divdamp_o2)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        z_graddiv_vn = random_field(grid, EdgeDim, KDim)
        vn = random_field(grid, EdgeDim, KDim)
        scal_divdamp_o2 = 5.0

        return dict(
            z_graddiv_vn=z_graddiv_vn,
            vn=vn,
            scal_divdamp_o2=scal_divdamp_o2,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
