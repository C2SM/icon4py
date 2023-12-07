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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_27 import (
    mo_solve_nonhydro_stencil_27,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil27(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_27
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        grid,
        scal_divdamp: np.array,
        bdy_divdamp: np.array,
        nudgecoeff_e: np.array,
        z_graddiv2_vn: np.array,
        vn: np.array,
        **kwargs,
    ) -> dict:
        nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
        vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        scal_divdamp = random_field(grid, KDim, dtype=wpfloat)
        bdy_divdamp = random_field(grid, KDim, dtype=wpfloat)
        nudgecoeff_e = random_field(grid, EdgeDim, dtype=wpfloat)
        z_graddiv2_vn = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
