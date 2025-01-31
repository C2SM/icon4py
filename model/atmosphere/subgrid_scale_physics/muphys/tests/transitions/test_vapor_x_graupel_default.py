# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_graupel
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestVaporXGraupelDefault(StencilTest):
    PROGRAM = vapor_x_graupel
    OUTPUTS = ("exchange_rate",)

    @staticmethod
    def reference(grid, t: np.array, p: np.array, rho: np.array, qg: np.array, dvsw: np.array, dvsi: np.array, dvsw0: np.array, dt: wpfloat, **kwargs) -> dict:
        return dict(exchange_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 278.026, dims.CellDim, dims.KDim, dtype=wpfloat),
            p                = constant_field(grid, 95987.1, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho              = constant_field(grid, 1.20041, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg               = constant_field(grid, 2.056496e-16, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw             = constant_field(grid, -0.00234674, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsi             = constant_field(grid, -0.00261576, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0            = constant_field(grid, -0.00076851, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt               = 30.0,
            exchange_rate    = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

