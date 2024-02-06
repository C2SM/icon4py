# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.set_two_edge_kdim_fields_to_zero_wp import (
    set_two_edge_kdim_fields_to_zero_wp,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestMoSolveNonhydroStencil14(StencilTest):
    PROGRAM = set_two_edge_kdim_fields_to_zero_wp
    OUTPUTS = ("edge_kdim_field_to_zero_wp_1", "edge_kdim_field_to_zero_wp_2")

    @staticmethod
    def reference(
        grid,
        edge_kdim_field_to_zero_wp_1: np.array,
        edge_kdim_field_to_zero_wp_2: np.array,
        **kwargs,
    ) -> dict:
        edge_kdim_field_to_zero_wp_1 = np.zeros_like(edge_kdim_field_to_zero_wp_1)
        edge_kdim_field_to_zero_wp_2 = np.zeros_like(edge_kdim_field_to_zero_wp_2)
        return dict(
            edge_kdim_field_to_zero_wp_1=edge_kdim_field_to_zero_wp_1,
            edge_kdim_field_to_zero_wp_2=edge_kdim_field_to_zero_wp_2,
        )

    @pytest.fixture
    def input_data(self, grid):
        edge_kdim_field_to_zero_wp_1 = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)
        edge_kdim_field_to_zero_wp_2 = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            edge_kdim_field_to_zero_wp_1=edge_kdim_field_to_zero_wp_1,
            edge_kdim_field_to_zero_wp_2=edge_kdim_field_to_zero_wp_2,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
