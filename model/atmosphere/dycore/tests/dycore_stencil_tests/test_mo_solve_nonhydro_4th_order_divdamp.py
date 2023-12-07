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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydro4thOrderDivdamp(StencilTest):
    PROGRAM = mo_solve_nonhydro_4th_order_divdamp
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        grid,
        scal_divdamp: np.array,
        z_graddiv2_vn: np.array,
        vn: np.array,
        **kwargs,
    ) -> dict:
        scal_divdamp = np.expand_dims(scal_divdamp, axis=0)
        vn = vn + (scal_divdamp * z_graddiv2_vn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        scal_divdamp = random_field(grid, KDim, dtype=wpfloat)
        z_graddiv2_vn = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            scal_divdamp=scal_divdamp,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
