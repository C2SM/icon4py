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

from icon4py.model.atmosphere.dycore.init_two_edge_kdim_fields_with_zero_wp import (
    init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestInitTwoEdgeKdimFieldsWithZeroWp(StencilTest):
    PROGRAM = init_two_edge_kdim_fields_with_zero_wp
    OUTPUTS = ("edge_kdim_field_with_zero_wp_1", "edge_kdim_field_with_zero_wp_2")

    @staticmethod
    def reference(
        grid,
        edge_kdim_field_with_zero_wp_1: np.array,
        edge_kdim_field_with_zero_wp_2: np.array,
        **kwargs,
    ) -> dict:
        edge_kdim_field_with_zero_wp_1 = np.zeros_like(edge_kdim_field_with_zero_wp_1)
        edge_kdim_field_with_zero_wp_2 = np.zeros_like(edge_kdim_field_with_zero_wp_2)
        return dict(
            edge_kdim_field_with_zero_wp_1=edge_kdim_field_with_zero_wp_1,
            edge_kdim_field_with_zero_wp_2=edge_kdim_field_with_zero_wp_2,
        )

    @pytest.fixture
    def input_data(self, grid):
        edge_kdim_field_with_zero_wp_1 = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        edge_kdim_field_with_zero_wp_2 = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            edge_kdim_field_with_zero_wp_1=edge_kdim_field_with_zero_wp_1,
            edge_kdim_field_with_zero_wp_2=edge_kdim_field_with_zero_wp_2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
