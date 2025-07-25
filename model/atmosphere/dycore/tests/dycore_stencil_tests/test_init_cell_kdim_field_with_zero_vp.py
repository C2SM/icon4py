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

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_vp import (
    init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


def init_cell_kdim_field_with_zero_vp_numpy(field_with_zero_vp: np.ndarray) -> np.ndarray:
    field_with_zero_vp = np.zeros_like(field_with_zero_vp)
    return field_with_zero_vp


class TestInitCellKdimFieldWithZeroVp(StencilTest):
    PROGRAM = init_cell_kdim_field_with_zero_vp
    OUTPUTS = ("field_with_zero_vp",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        field_with_zero_vp: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        field_with_zero_vp = init_cell_kdim_field_with_zero_vp_numpy(
            field_with_zero_vp=field_with_zero_vp
        )
        return dict(field_with_zero_vp=field_with_zero_vp)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        field_with_zero_vp = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            field_with_zero_vp=field_with_zero_vp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
