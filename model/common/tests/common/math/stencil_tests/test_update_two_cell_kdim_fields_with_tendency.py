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

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.math.stencils.update_two_cell_kdim_fields_with_tendency import (
    update_two_cell_kdim_fields_with_tendency,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestUpdateTwoCellKdimFieldsWithTendency(stencil_tests.StencilTest):
    PROGRAM = update_two_cell_kdim_fields_with_tendency
    OUTPUTS = ("new_field_1", "new_field_2")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        field_1: np.ndarray,
        field_2: np.ndarray,
        tendency_1: np.ndarray,
        tendency_2: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        new_field_1 = np.zeros_like(field_1)
        new_field_2 = np.zeros_like(field_2)
        hs, he = horizontal_start, horizontal_end
        vs, ve = vertical_start, vertical_end
        new_field_1[hs:he, vs:ve] = field_1[hs:he, vs:ve] + tendency_1[hs:he, vs:ve] * dtime
        new_field_2[hs:he, vs:ve] = field_2[hs:he, vs:ve] + tendency_2[hs:he, vs:ve] * dtime
        return dict(new_field_1=new_field_1, new_field_2=new_field_2)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        field_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        field_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tendency_1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tendency_2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        new_field_1 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        new_field_2 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        # exercise a partial horizontal domain (the tmx wind-update loop bounds)
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            field_1=field_1,
            field_2=field_2,
            tendency_1=tendency_1,
            tendency_2=tendency_2,
            new_field_1=new_field_1,
            new_field_2=new_field_2,
            dtime=ta.wpfloat(300.0),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
