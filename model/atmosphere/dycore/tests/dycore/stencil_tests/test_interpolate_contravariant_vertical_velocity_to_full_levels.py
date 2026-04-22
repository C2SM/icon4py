# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.testing import stencil_tests


def interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
    z_w_con_c: np.ndarray,
) -> np.ndarray:
    z_w_con_c_full = 0.5 * (z_w_con_c[:, :-1] + z_w_con_c[:, 1:])
    return z_w_con_c_full


class TestInterpolateContravariantVerticalVelocityToFullLevels(stencil_tests.StencilTest):
    PROGRAM = interpolate_contravariant_vertical_velocity_to_full_levels
    OUTPUTS = ("z_w_con_c_full",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, z_w_con_c: np.ndarray, **kwargs: Any) -> dict:
        z_w_con_c_full = interpolate_contravariant_vertical_velocity_to_full_levels_numpy(z_w_con_c)
        return dict(z_w_con_c_full=z_w_con_c_full)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_w_con_c = self.data_alloc.random_field(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat
        )

        z_w_con_c_full = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            z_w_con_c=z_w_con_c,
            z_w_con_c_full=z_w_con_c_full,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
