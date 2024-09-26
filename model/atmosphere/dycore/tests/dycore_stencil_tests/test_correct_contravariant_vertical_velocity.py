# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.dycore.correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat


def correct_contravariant_vertical_velocity_numpy(
    w_concorr_c: np.array, z_w_con_c: np.array
) -> np.array:
    z_w_con_c = z_w_con_c - w_concorr_c
    return z_w_con_c


class TestCorrectContravariantVerticalVelocity(StencilTest):
    PROGRAM = correct_contravariant_vertical_velocity
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(grid, w_concorr_c: np.array, z_w_con_c: np.array, **kwargs) -> dict:
        z_w_con_c = correct_contravariant_vertical_velocity_numpy(w_concorr_c, z_w_con_c)
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        w_concorr_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w_concorr_c=w_concorr_c,
            z_w_con_c=z_w_con_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
