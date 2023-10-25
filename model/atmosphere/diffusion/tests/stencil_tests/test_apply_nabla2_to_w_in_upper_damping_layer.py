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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestApplyNabla2ToWInUpperDampingLayer(StencilTest):
    PROGRAM = apply_nabla2_to_w_in_upper_damping_layer
    OUTPUTS = ("w",)

    @pytest.fixture
    def input_data(self, grid):
        w = random_field(grid, CellDim, KDim)
        diff_multfac_n2w = random_field(grid, KDim)
        cell_area = random_field(grid, CellDim)
        z_nabla2_c = random_field(grid, CellDim, KDim)

        return dict(
            w=w,
            diff_multfac_n2w=diff_multfac_n2w,
            cell_area=cell_area,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=int32(0),
            horizontal_end=int(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        grid,
        w: np.array,
        diff_multfac_n2w: np.array,
        cell_area: np.array,
        z_nabla2_c: np.array,
        **kwargs,
    ) -> np.array:
        cell_area = np.expand_dims(cell_area, axis=-1)
        w = w + diff_multfac_n2w * cell_area * z_nabla2_c
        return dict(w=w)
