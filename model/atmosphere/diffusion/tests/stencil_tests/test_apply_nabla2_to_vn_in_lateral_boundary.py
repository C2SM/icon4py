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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestApplyNabla2ToVnInLateralBoundary(StencilTest):
    PROGRAM = apply_nabla2_to_vn_in_lateral_boundary
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid):
        fac_bdydiff_v = 5.0
        z_nabla2_e = random_field(grid, EdgeDim, KDim)
        area_edge = random_field(grid, EdgeDim)
        vn = random_field(grid, EdgeDim, KDim)
        return dict(
            fac_bdydiff_v=fac_bdydiff_v,
            z_nabla2_e=z_nabla2_e,
            area_edge=area_edge,
            vn=vn,
        )

    @staticmethod
    def reference(
        grid, z_nabla2_e: np.array, area_edge: np.array, vn: np.array, fac_bdydiff_v
    ) -> np.array:
        area_edge = np.expand_dims(area_edge, axis=-1)
        vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
        return dict(vn=vn)
