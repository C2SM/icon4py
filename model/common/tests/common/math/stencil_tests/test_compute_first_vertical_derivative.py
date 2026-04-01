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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.derivative import compute_first_vertical_derivative_at_cells
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.testing import stencil_tests


def compute_first_vertical_derivative_numpy(
    cell_kdim_field: np.ndarray, inv_ddqz_z_full: np.ndarray
) -> np.ndarray:
    first_vertical_derivative = (cell_kdim_field[:, :-1] - cell_kdim_field[:, 1:]) * inv_ddqz_z_full
    return first_vertical_derivative


class TestComputeFirstVerticalDerivative(stencil_tests.StencilTest):
    PROGRAM = compute_first_vertical_derivative_at_cells
    OUTPUTS = ("first_vertical_derivative",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        cell_kdim_field: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        first_vertical_derivative = compute_first_vertical_derivative_numpy(
            cell_kdim_field, inv_ddqz_z_full
        )
        return dict(first_vertical_derivative=first_vertical_derivative)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        cell_kdim_field = self.data_alloc.random_field(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat
        )
        inv_ddqz_z_full = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        first_vertical_derivative = self.data_alloc.zero_field(
            dims.CellDim, dims.KDim, dtype=vpfloat
        )

        return dict(
            cell_kdim_field=cell_kdim_field,
            inv_ddqz_z_full=inv_ddqz_z_full,
            first_vertical_derivative=first_vertical_derivative,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
