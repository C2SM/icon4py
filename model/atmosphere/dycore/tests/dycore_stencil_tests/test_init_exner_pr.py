# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.init_exner_pr import (
    init_exner_pr,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


class TestInitExnerPr(StencilTest):
    PROGRAM = init_exner_pr
    OUTPUTS = ("exner_pr",)

    @staticmethod
    def reference(grid, exner: np.array, exner_ref: np.array, **kwargs) -> dict:
        exner_pr = exner - exner_ref
        return dict(
            exner_pr=exner_pr,
        )

    @pytest.fixture
    def input_data(self, grid):
        exner = random_field(grid, CellDim, KDim, dtype=vpfloat)
        exner_ref = random_field(grid, CellDim, KDim, dtype=vpfloat)
        exner_pr = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            exner=exner,
            exner_ref=exner_ref,
            exner_pr=exner_pr,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
