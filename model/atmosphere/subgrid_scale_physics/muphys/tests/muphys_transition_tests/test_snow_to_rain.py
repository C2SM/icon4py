# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.snow_to_rain import snow_to_rain
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestSnowToRainDefault(StencilTest):
    PROGRAM = snow_to_rain
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, p: np.array, rho: np.array, dvsw0: np.array, qs: np.array, QMIN: wpfloat, TX: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 3.7268547760462804e-07))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 275.83, dims.CellDim, dtype=wpfloat),
            p                = constant_field(grid, 80134.5, dims.CellDim, dtype=wpfloat),
            rho              = constant_field(grid, 1.04892, dims.CellDim, dtype=wpfloat),
            dvsw0            = constant_field(grid, 0.00258631, dims.CellDim, dtype=wpfloat),
            qs               = constant_field(grid, 1.47687e-6, dims.CellDim, dtype=wpfloat),
            QMIN             = graupel_ct.qmin,
            TX               = graupel_ct.tx,
            TMELT            = thermodyn.tmelt,
            conversion_rate  = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )

