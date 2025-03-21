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

from icon4py.model.atmosphere.dycore.stencils.correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


def correct_contravariant_vertical_velocity_numpy(
    z_w_con_c: np.ndarray, w_concorr_c: np.ndarray
) -> np.ndarray:
    z_w_con_c = z_w_con_c - w_concorr_c
    return z_w_con_c


class TestCorrectContravariantVerticalVelocity(StencilTest):
    PROGRAM = correct_contravariant_vertical_velocity
    OUTPUTS = ("z_w_con_c",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        w_concorr_c: np.ndarray,
        z_w_con_c: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_w_con_c = correct_contravariant_vertical_velocity_numpy(z_w_con_c, w_concorr_c)
        return dict(z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
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
