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

from icon4py.model.atmosphere.advection.rbf_intp_edge_stencil_01 import rbf_intp_edge_stencil_01
from icon4py.model.common.dimension import E2C2EDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestRbfIntpEdgeStencil01(StencilTest):
    PROGRAM = rbf_intp_edge_stencil_01
    OUTPUTS = ("p_vt_out",)

    @staticmethod
    def reference(grid, p_vn_in: np.array, ptr_coeff: np.array, **kwargs) -> np.array:
        ptr_coeff = np.expand_dims(ptr_coeff, axis=-1)
        p_vt_out = np.sum(p_vn_in[grid.connectivities[E2C2EDim]] * ptr_coeff, axis=1)
        return dict(p_vt_out=p_vt_out)

    @pytest.fixture
    def input_data(self, grid):
        p_vn_in = random_field(grid, EdgeDim, KDim)
        ptr_coeff = random_field(grid, EdgeDim, E2C2EDim)
        p_vt_out = zero_field(grid, EdgeDim, KDim)
        return dict(p_vn_in=p_vn_in, ptr_coeff=ptr_coeff, p_vt_out=p_vt_out)
