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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_38 import (
    mo_solve_nonhydro_stencil_38,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoSolveNonhydroStencil38(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_38
    OUTPUTS = ("vn_ie",)

    @staticmethod
    def reference(grid, vn: np.array, wgtfacq_e: np.array, **kwargs) -> np.array:
        vn_ie = np.zeros_like(vn)
        vn_ie[:, -1] = (
            np.roll(wgtfacq_e, shift=1, axis=1) * np.roll(vn, shift=1, axis=1)
            + np.roll(wgtfacq_e, shift=2, axis=1) * np.roll(vn, shift=2, axis=1)
            + np.roll(wgtfacq_e, shift=3, axis=1) * np.roll(vn, shift=3, axis=1)
        )[:, -1]
        return dict(vn_ie=vn_ie)

    @pytest.fixture
    def input_data(self, grid):
        wgtfacq_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vn_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            wgtfacq_e=wgtfacq_e,
            vn_ie=vn_ie,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(grid.num_levels - 1),
            vertical_end=int32(grid.num_levels),
        )
