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

from icon4py.model.atmosphere.dycore.stencils.correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


def correct_contravariant_vertical_velocity_numpy(
    w_concorr_c: np.array, z_w_con_c: np.array
) -> np.array:
    z_w_con_c = z_w_con_c - w_concorr_c
    return z_w_con_c


class TestCorrectContravariantVerticalVelocity(StencilTest):
    PROGRAM = correct_contravariant_vertical_velocity
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(grid, w_concorr_c: np.array, z_w_con_c: np.array, **kwargs) -> dict:
        z_w_con_c = correct_contravariant_vertical_velocity_numpy(w_concorr_c, z_w_con_c)
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        w_concorr_c = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            w_concorr_c=w_concorr_c,
            z_w_con_c=z_w_con_c,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
