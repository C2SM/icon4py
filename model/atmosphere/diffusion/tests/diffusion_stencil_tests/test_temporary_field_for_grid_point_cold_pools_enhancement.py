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

from icon4py.model.atmosphere.diffusion.stencils.temporary_field_for_grid_point_cold_pools_enhancement import (
    temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common.dimension import C2E2CDim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestTemporaryFieldForGridPointColdPoolsEnhancement(StencilTest):
    PROGRAM = temporary_field_for_grid_point_cold_pools_enhancement
    OUTPUTS = ("enh_diffu_3d",)

    @staticmethod
    def reference(
        grid, theta_v: np.array, theta_ref_mc: np.array, thresh_tdiff, smallest_vpfloat, **kwargs
    ) -> dict:
        c2e2c = grid.connectivities[C2E2CDim]
        tdiff = (
            theta_v
            - np.sum(np.where((c2e2c != -1)[:, :, np.newaxis], theta_v[c2e2c], 0), axis=1) / 3
        )
        trefdiff = (
            theta_ref_mc
            - np.sum(
                np.where((c2e2c != -1)[:, :, np.newaxis], theta_ref_mc[c2e2c], 0),
                axis=1,
            )
            / 3
        )

        enh_diffu_3d = np.where(
            ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0)
            | (tdiff - trefdiff < 1.5 * thresh_tdiff),
            (thresh_tdiff - tdiff + trefdiff) * 5e-4,
            smallest_vpfloat,
        )

        return dict(enh_diffu_3d=enh_diffu_3d)

    @pytest.fixture
    def input_data(self, grid):
        theta_v = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_ref_mc = random_field(grid, CellDim, KDim, dtype=vpfloat)
        enh_diffu_3d = zero_field(grid, CellDim, KDim, dtype=vpfloat)
        thresh_tdiff = wpfloat("5.0")
        smallest_vpfloat = -np.finfo(vpfloat).max

        return dict(
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            enh_diffu_3d=enh_diffu_3d,
            thresh_tdiff=thresh_tdiff,
            smallest_vpfloat=smallest_vpfloat,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
