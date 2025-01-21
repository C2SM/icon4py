# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.dqsatdT_rho import dqsatdT_rho
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestQsatRho(StencilTest):
    PROGRAM = dqsatdT_rho
    OUTPUTS = ("derivative",)

    @staticmethod
    def reference(grid, qs: np.array, t: np.array, TMELT: wpfloat, **kwargs) -> dict:
        return dict(derivative=np.full(t.shape, 0.00030825070286492049))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qs               = constant_field(grid, 0.00448941, dims.CellDim, dtype=wpfloat),
            t                = constant_field(grid, 273.909, dims.CellDim, dtype=wpfloat),
            TMELT            = thermodyn.tmelt,
            derivative       = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
