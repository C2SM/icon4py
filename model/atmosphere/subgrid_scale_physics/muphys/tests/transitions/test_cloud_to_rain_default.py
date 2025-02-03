# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_rain
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat

class TestCloudToRainDefault(StencilTest):
    PROGRAM = cloud_to_rain
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qr: np.array, nc: np.array, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t                   = constant_field(grid, 281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc                  = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr                  = constant_field(grid, 52312e-7, dims.CellDim, dims.KDim, dtype=wpfloat),
            nc                  = 100.0,
            conversion_rate     = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

