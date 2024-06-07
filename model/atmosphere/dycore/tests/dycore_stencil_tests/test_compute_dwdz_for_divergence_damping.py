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

from icon4py.model.atmosphere.dycore.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeDwdzForDivergenceDamping(StencilTest):
    PROGRAM = compute_dwdz_for_divergence_damping
    OUTPUTS = ("z_dwdz_dd",)

    @staticmethod
    def reference(
        grid, inv_ddqz_z_full: np.array, w: np.array, w_concorr_c: np.array, **kwargs
    ) -> dict:
        z_dwdz_dd = inv_ddqz_z_full * (
            (w[:, :-1] - w[:, 1:]) - (w_concorr_c[:, :-1] - w_concorr_c[:, 1:])
        )
        return dict(z_dwdz_dd=z_dwdz_dd)

    @pytest.fixture
    def input_data(self, grid):
        inv_ddqz_z_full = random_field(grid, CellDim, KDim, dtype=vpfloat)
        w = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=wpfloat)
        w_concorr_c = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=vpfloat)
        z_dwdz_dd = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            inv_ddqz_z_full=inv_ddqz_z_full,
            w=w,
            w_concorr_c=w_concorr_c,
            z_dwdz_dd=z_dwdz_dd,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
