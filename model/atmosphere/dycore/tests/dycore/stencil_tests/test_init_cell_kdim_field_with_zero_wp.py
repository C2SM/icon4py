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

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import zero_field
from icon4py.model.testing.helpers import StencilTest


class TestInitCellKdimFieldWithZeroWp(StencilTest):
    PROGRAM = init_cell_kdim_field_with_zero_wp
    OUTPUTS = ("field_with_zero_wp",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        field_with_zero_wp: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        field_with_zero_wp = np.zeros_like(field_with_zero_wp)
        return dict(field_with_zero_wp=field_with_zero_wp)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        field_with_zero_wp = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            field_with_zero_wp=field_with_zero_wp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
