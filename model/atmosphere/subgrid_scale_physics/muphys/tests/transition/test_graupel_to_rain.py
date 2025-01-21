# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.graupel_to_rain import graupel_to_rain
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestGraupelToRain(StencilTest):
    PROGRAM = graupel_to_rain
    OUTPUTS = ("rain_rate",)

    @staticmethod
    def reference(grid, t: np.array, p: np.array, rho: np.array, dvsw0: np.array, qg: np.array, QMIN: wpfloat, TX: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(rain_rate=np.full(t.shape, 5.9748142538569357e-13))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t       = constant_field(grid, 280.156, dims.CellDim, dtype=wpfloat),
            p       = constant_field(grid, 98889.4, dims.CellDim, dtype=wpfloat),
            rho     = constant_field(grid, 1.22804, dims.CellDim, dtype=wpfloat),
            dvsw0   = constant_field(grid, -0.00167867, dims.CellDim, dtype=wpfloat),
            qg      = constant_field(grid, 1.53968e-15, dims.CellDim, dtype=wpfloat),
            QMIN    = graupel_ct.qmin,
            TX      = graupel_ct.tx,
            TMELT   = thermodyn.tmelt,
            rain_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )

