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
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.testing import helpers


def interpolate_to_cell_center_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    interpolant: np.ndarray,
    e_bln_c_s: np.ndarray,
    **kwargs,
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2e = connectivities[dims.C2EDim]
    c2ce = helpers.as_1d_connectivity(c2e)

    interpolation = np.sum(
        interpolant[c2e] * e_bln_c_s[c2ce],
        axis=1,
    )
    return interpolation


class TestInterpolateToCellCenter(helpers.StencilTest):
    PROGRAM = interpolate_to_cell_center
    OUTPUTS = ("interpolation",)

    @staticmethod
    def reference(grid, interpolant: np.array, e_bln_c_s: np.array, **kwargs) -> dict:
        interpolation = interpolate_to_cell_center_numpy(grid, interpolant, e_bln_c_s)
        return dict(interpolation=interpolation)

    @pytest.fixture
    def input_data(self, grid):
        interpolant = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim, dtype=ta.wpfloat)
        interpolation = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            interpolant=interpolant,
            e_bln_c_s=e_bln_c_s,
            interpolation=interpolation,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
