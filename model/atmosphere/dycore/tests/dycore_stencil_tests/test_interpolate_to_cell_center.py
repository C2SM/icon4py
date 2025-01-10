# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_cell_center import (
    interpolate_to_cell_center,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.helpers import StencilTest


def interpolate_to_cell_center_numpy(
    grid, interpolant: np.array, e_bln_c_s: np.array, **kwargs
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2ce = grid.get_offset_provider("C2CE").ndarray

    interpolation = np.sum(
        interpolant[grid.connectivities[dims.C2EDim]] * e_bln_c_s[c2ce],
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
        interpolant = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat)
        interpolation = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            interpolant=interpolant,
            e_bln_c_s=data_alloc.as_1D_sparse_field(e_bln_c_s, dims.CEDim),
            interpolation=interpolation,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
