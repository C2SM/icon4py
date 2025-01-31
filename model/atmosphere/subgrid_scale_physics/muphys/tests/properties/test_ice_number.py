# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import ice_number

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestIceNumber(StencilTest):
    PROGRAM = ice_number
    OUTPUTS = ("ice_number",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, **kwargs) -> dict:
        return dict(ice_number=np.full(t.shape, 5.0507995893464388))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t               = constant_field(grid, 272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho             = constant_field(grid, 1.12442, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_number      = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
