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

from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    interpolate_to_half_levels_vp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat

def interpolate_to_half_levels_vp_numpy(grid, wgtfac_c: np.array, interpolant: np.array) -> np.array:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_vp = (
        wgtfac_c * interpolant + (1.0 - wgtfac_c) * interpolant_offset_1
    )
    interpolation_to_half_levels_vp[:, 0] = 0

    return interpolation_to_half_levels_vp

class TestMoSolveNonhydroStencil05(StencilTest):
    PROGRAM = interpolate_to_half_levels_vp
    OUTPUTS = ("interpolation_to_half_levels_vp",)

    @staticmethod
    def reference(grid, wgtfac_c: np.array, interpolant: np.array, **kwargs) -> dict:
        interpolation_to_half_levels_vp = interpolate_to_half_levels_vp_numpy(grid=grid, wgtfac_c=wgtfac_c, interpolant=interpolant)
        return dict(interpolation_to_half_levels_vp=interpolation_to_half_levels_vp)

    @pytest.fixture
    def input_data(self, grid):
        interpolant = random_field(grid, CellDim, KDim, dtype=vpfloat)
        wgtfac_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        interpolation_to_half_levels_vp = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            interpolant=interpolant,
            interpolation_to_half_levels_vp=interpolation_to_half_levels_vp,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
