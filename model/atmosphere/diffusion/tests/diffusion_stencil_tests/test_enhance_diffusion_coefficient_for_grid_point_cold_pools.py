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
import math

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


class TestEnhanceDiffusionCoefficientForGridPointColdPools(StencilTest):
    PROGRAM = enhance_diffusion_coefficient_for_grid_point_cold_pools
    OUTPUTS = ("kh_smag_e",)

    @staticmethod
    def reference(grid, kh_smag_e: np.array, enh_diffu_3d: np.array, **kwargs) -> np.array:
        e2c = grid.connectivities[E2CDim]
        kh_smag_e = np.maximum(
            kh_smag_e,
            np.max(np.where((e2c != -1)[:, :, np.newaxis], enh_diffu_3d[e2c], -math.inf), axis=1),
        )
        return dict(kh_smag_e=kh_smag_e)

    @pytest.fixture
    def input_data(self, grid):
        kh_smag_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        enh_diffu_3d = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            kh_smag_e=kh_smag_e,
            enh_diffu_3d=enh_diffu_3d,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
