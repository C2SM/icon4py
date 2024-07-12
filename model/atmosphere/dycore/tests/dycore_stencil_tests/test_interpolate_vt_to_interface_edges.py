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

from icon4py.model.atmosphere.dycore.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


def interpolate_vt_to_interface_edges_numpy(
    grid, wgtfac_e: np.array, vt: np.array, **kwargs
) -> np.array:
    vt_k_minus_1 = np.roll(vt, shift=1, axis=1)
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_k_minus_1
    z_vt_ie[:, 0] = 0
    return z_vt_ie


class TestInterpolateVtToInterfaceEdges(StencilTest):
    PROGRAM = interpolate_vt_to_interface_edges
    OUTPUTS = ("z_vt_ie",)

    @staticmethod
    def reference(
        grid,
        wgtfac_e: np.array,
        vt: np.array,
        z_vt_ie: np.array,
        horizontal_start: int32,
        horizontal_end: int32,
        vertical_start: int32,
        vertical_end: int32,
    ) -> dict:
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        z_vt_ie = z_vt_ie.copy()
        z_vt_ie[subset] = interpolate_vt_to_interface_edges_numpy(grid, wgtfac_e, vt)[subset]
        return dict(z_vt_ie=z_vt_ie)

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        z_vt_ie = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vt=vt,
            z_vt_ie=z_vt_ie,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=1,
            vertical_end=int32(grid.num_levels),
        )
