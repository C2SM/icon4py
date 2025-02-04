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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import vel_scale_factor_ice

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestVelScaleFactorIce(StencilTest):
    PROGRAM = vel_scale_factor_ice
    OUTPUTS = ("scale_factor",)

    @staticmethod
    def reference(grid, xrho: np.array, **kwargs) -> dict:
        return dict(scale_factor=np.full(xrho.shape, 1.1158596098981044))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            xrho            = constant_field(grid, 1.17873, dims.CellDim, dims.KDim, dtype=wpfloat),
            scale_factor    = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
