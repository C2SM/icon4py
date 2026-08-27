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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    interpolate_km_to_full_level_cells,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def interpolate_km_to_full_level_cells_numpy(km_ic: np.ndarray, *, km_min: float) -> np.ndarray:
    return np.maximum(km_min, 0.5 * (km_ic[:, :-1] + km_ic[:, 1:]))


class TestInterpolateKmToFullLevelCells(stencil_tests.StencilTest):
    PROGRAM = interpolate_km_to_full_level_cells
    OUTPUTS = ("km_c",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        km_ic: np.ndarray,
        km_min: float,
        **kwargs,
    ) -> dict:
        km_c = interpolate_km_to_full_level_cells_numpy(km_ic, km_min=km_min)
        return dict(km_c=km_c)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        km_ic = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=wpfloat, extend={dims.KDim: 1}
        )
        km_c = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            km_ic=km_ic,
            km_c=km_c,
            # large enough that the floor is active for part of the field
            km_min=wpfloat(0.5),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
