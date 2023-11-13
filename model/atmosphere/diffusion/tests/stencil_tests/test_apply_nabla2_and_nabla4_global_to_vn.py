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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestApplyNabla2AndNabla4GlobalToVn(StencilTest):
    PROGRAM = apply_nabla2_and_nabla4_global_to_vn
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid):
        area_edge = random_field(grid, EdgeDim, dtype=wpfloat))
        kh_smag_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_nabla2_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_nabla4_e2 = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        diff_multfac_vn = random_field(grid, KDim, dtype=wpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            z_nabla2_e=z_nabla2_e,
            z_nabla4_e2=z_nabla4_e2,
            diff_multfac_vn=diff_multfac_vn,
            vn=vn,
        )

    @staticmethod
    def reference(grid, area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn):
        area_edge = np.expand_dims(area_edge, axis=-1)
        diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
        vn = vn + area_edge * (kh_smag_e * z_nabla2_e - diff_multfac_vn * z_nabla4_e2 * area_edge)
        return dict(vn=vn)
