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

from icon4py.model.atmosphere.dycore.interpolate_to_cell_center import interpolate_to_cell_center
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


def interpolate_to_cell_center_numpy(
    grid, interpolant: np.array, e_bln_c_s: np.array, **kwargs
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)

    if grid.config.on_gpu:
        connectivity = grid.get_offset_provider("C2CE").table.get()
    else:
        connectivity = grid.get_offset_provider("C2CE").table

    interpolation = np.sum(
        interpolant[grid.connectivities[C2EDim]]
        * e_bln_c_s[connectivity],
        axis=1,
    )
    return interpolation


class TestInterpolateToCellCenter(StencilTest):
    PROGRAM = interpolate_to_cell_center
    OUTPUTS = ("interpolation",)

    @staticmethod
    def reference(grid, interpolant: np.array, e_bln_c_s: np.array, **kwargs) -> dict:
        interpolation = interpolate_to_cell_center_numpy(grid, interpolant, e_bln_c_s)
        return dict(interpolation=interpolation)

    @pytest.fixture
    def input_data(self, grid):
        interpolant = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        e_bln_c_s = random_field(grid, CellDim, C2EDim, dtype=wpfloat)
        interpolation = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            interpolant=interpolant,
            e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
            interpolation=interpolation,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
