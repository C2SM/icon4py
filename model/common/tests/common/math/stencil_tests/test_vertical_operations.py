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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.vertical_operations import (
    set_cell_field_on_model_levels_on_cells_wp,
    set_constant_on_model_levels_on_cells_wp,
)
from icon4py.model.testing import stencil_tests


class TestSetConstantOnModelLevelsOnCellsWp(stencil_tests.StencilTest):
    """Only the given cells and levels of `field` take the constant."""

    PROGRAM = set_constant_on_model_levels_on_cells_wp
    OUTPUTS = ("field",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        field: np.ndarray,
        value: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        expected = field.copy()
        expected[horizontal_start:horizontal_end, vertical_start:vertical_end] = value
        return dict(field=expected)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            field=data_alloc.random_field(dims.CellDim, dims.KDim),
            value=2.5,
            horizontal_start=1,
            horizontal_end=gtx.int32(grid.num_cells - 1),
            # the lowest model level
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestSetCellFieldOnModelLevelsOnCellsWp(stencil_tests.StencilTest):
    """Only the given cells and levels of `field` take the cell field."""

    PROGRAM = set_cell_field_on_model_levels_on_cells_wp
    OUTPUTS = ("field",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        field: np.ndarray,
        value: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        expected = field.copy()
        cells = slice(horizontal_start, horizontal_end)
        expected[cells, vertical_start:vertical_end] = value[cells, np.newaxis]
        return dict(field=expected)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            field=data_alloc.random_field(dims.CellDim, dims.KDim),
            value=data_alloc.random_field(dims.CellDim),
            horizontal_start=1,
            horizontal_end=gtx.int32(grid.num_cells - 1),
            # the lowest model level
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )
