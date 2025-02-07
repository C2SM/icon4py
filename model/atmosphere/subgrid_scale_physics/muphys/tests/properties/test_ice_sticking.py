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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import ice_sticking
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

class TestIceSticking(StencilTest):
    PROGRAM = ice_sticking
    OUTPUTS = ("ice_sticking",)

    @staticmethod
    def reference(grid, t: np.array, **kwargs) -> dict:
        return dict(ice_sticking=np.full(t.shape, 0.8697930232044021))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t            = data_alloc.constant_field(grid, 271.6, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_sticking = data_alloc.constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
