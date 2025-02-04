# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import rain_to_graupel
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestRainToGraupelDefault(StencilTest):
    PROGRAM = rain_to_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qr: np.array, qi: np.array, qs: np.array, mi: np.array, dvsw: np.array, dt: wpfloat, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t       = constant_field(grid, 272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho     = constant_field(grid, 1.12442, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc      = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr      = constant_field(grid, 1.34006e-17, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi      = constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs      = constant_field(grid, 1.02627e-19, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi      = constant_field(grid, 1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw    = constant_field(grid, -0.000635669, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt      = 30.0,
            conversion_rate = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )
