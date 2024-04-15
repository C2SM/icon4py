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

from icon4py.model.atmosphere.dycore.compute_contravariant_correction import (
    compute_contravariant_correction,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_contravariant_correction_numpy(
    vn: np.array, ddxn_z_full: np.array, ddxt_z_full: np.array, vt: np.array
) -> np.array:
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


class TestComputeContravariantCorrection(StencilTest):
    PROGRAM = compute_contravariant_correction
    OUTPUTS = ("z_w_concorr_me",)

    @staticmethod
    def reference(
        grid,
        vn: np.array,
        ddxn_z_full: np.array,
        ddxt_z_full: np.array,
        vt: np.array,
        **kwargs,
    ) -> dict:
        z_w_concorr_me = compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, vt)
        return dict(z_w_concorr_me=z_w_concorr_me)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        ddxn_z_full = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        ddxt_z_full = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_w_concorr_me = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            vt=vt,
            z_w_concorr_me=z_w_concorr_me,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
