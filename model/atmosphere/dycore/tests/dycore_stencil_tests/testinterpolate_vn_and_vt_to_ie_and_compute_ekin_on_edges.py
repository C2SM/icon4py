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

from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestInterpolateVnAndVtToIeAndComputeEkinOnEdges(StencilTest):
    PROGRAM = interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(
        grid, wgtfac_e: np.array, vn: np.array, vt: np.array, **kwargs
    ) -> tuple[np.array, np.array]:
        vn_offset_1 = np.roll(vn, shift=1, axis=1)
        vt_offset_1 = np.roll(vt, shift=1, axis=1)

        vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_offset_1
        vn_ie[:, 0] = 0
        z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_offset_1
        z_vt_ie[:, 0] = 0
        z_kin_hor_e = 0.5 * (vn**2 + vt**2)
        z_kin_hor_e[:, 0] = 0
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_vt_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=1,
            vertical_end=int32(grid.num_levels),
        )
