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

from icon4py.model.atmosphere.dycore.extrapolate_at_top import extrapolate_at_top
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def extrapolate_at_top_numpy(grid, wgtfacq_e: np.array, vn: np.array) -> np.array:
    vn_k_minus_1 = vn[:, -1]
    vn_k_minus_2 = vn[:, -2]
    vn_k_minus_3 = vn[:, -3]
    wgtfacq_e_k_minus_1 = wgtfacq_e[:, -1]
    wgtfacq_e_k_minus_2 = wgtfacq_e[:, -2]
    wgtfacq_e_k_minus_3 = wgtfacq_e[:, -3]
    vn_ie = np.zeros((grid.num_edges, grid.num_levels + 1), dtype=vpfloat)
    vn_ie[:, -1] = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )
    return vn_ie


class TestExtrapolateAtTop(StencilTest):
    PROGRAM = extrapolate_at_top
    OUTPUTS = ("vn_ie",)

    @staticmethod
    def reference(grid, wgtfacq_e: np.array, vn: np.array, **kwargs) -> dict:
        vn_ie = extrapolate_at_top_numpy(grid, wgtfacq_e, vn)
        return dict(vn_ie=vn_ie)

    @pytest.fixture
    def input_data(self, grid):
        wgtfacq_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vn_ie = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat, extend={dims.KDim: 1})

        return dict(
            wgtfacq_e=wgtfacq_e,
            vn=vn,
            vn_ie=vn_ie,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(grid.num_levels),
            vertical_end=int32(grid.num_levels + 1),
        )
