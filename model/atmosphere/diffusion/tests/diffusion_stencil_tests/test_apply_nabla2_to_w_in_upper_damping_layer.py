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
from icon4py.model.common.type_alias import vpfloat, wpfloat


def apply_nabla2_to_w_in_upper_damping_layer_numpy(
    w: np.array,
    diff_multfac_n2w: np.array,
    cell_area: np.array,
    z_nabla2_c: np.array,
):
    cell_area = np.expand_dims(cell_area, axis=-1)
    diff_multfac_n2w_extend = np.insert(diff_multfac_n2w, diff_multfac_n2w.shape, 0.0, axis=0)
    w = w + diff_multfac_n2w_extend * cell_area * z_nabla2_c
    return w


class TestApplyNabla2ToWInUpperDampingLayer(StencilTest):
    PROGRAM = apply_nabla2_to_w_in_upper_damping_layer
    OUTPUTS = ("w",)

    @pytest.fixture
    def input_data(self, grid):
        w = random_field(grid, CellDim, KDim, dtype=wpfloat)
        diff_multfac_n2w = random_field(grid, KDim, dtype=wpfloat)
        cell_area = random_field(grid, CellDim, dtype=wpfloat)
        z_nabla2_c = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            w=w,
            diff_multfac_n2w=diff_multfac_n2w,
            cell_area=cell_area,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=int(grid.num_cells),
            vertical_start=0,
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
    ) -> dict:
        w = apply_nabla2_to_w_in_upper_damping_layer_numpy(
            w, diff_multfac_n2w, cell_area, z_nabla2_c
        )
        return dict(w=w)
