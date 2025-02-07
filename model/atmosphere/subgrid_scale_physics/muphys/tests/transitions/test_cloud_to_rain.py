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
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

class TestCloudToRain(StencilTest):
    PROGRAM = cloud_to_rain
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qr: np.array, nc: np.array, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 0.0045484481075162512))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t                   = data_alloc.constant_field(grid, 267.25, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc                  = data_alloc.constant_field(grid, 5.52921e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr                  = data_alloc.constant_field(grid, 2.01511e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            nc                  = 100.0, 
            conversion_rate     = data_alloc.constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

