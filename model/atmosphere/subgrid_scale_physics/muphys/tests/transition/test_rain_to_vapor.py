# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.rain_to_vapor import rain_to_vapor
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestRainToVapor(StencilTest):
    PROGRAM = rain_to_vapor
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qr: np.array, dvsw: np.array, dt: wpfloat, QMIN: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 2.8556697055499901e-19))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t       = constant_field(grid, 258.542, dims.CellDim, dtype=wpfloat),
            rho     = constant_field(grid, 0.956089, dims.CellDim, dtype=wpfloat),
            qc      = constant_field(grid, 0.0, dims.CellDim, dtype=wpfloat),
            qr      = constant_field(grid, 3.01332e-11, dims.CellDim, dtype=wpfloat),
            dvsw    = constant_field(grid, -1.0e-10, dims.CellDim, dtype=wpfloat),
            dt      = 30.0,
            QMIN    = graupel_ct.qmin,
            TMELT   = thermodyn.tmelt,
            conversion_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
