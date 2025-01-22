# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_ice
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestVaporXIceDefault(StencilTest):
    PROGRAM = vapor_x_ice
    OUTPUTS = ("vapor_deposition_rate",)

    @staticmethod
    def reference(grid, qi: np.array, mi: np.array, eta: np.array, dvsi: np.array, rho: np.array, dt: wpfloat, QMIN: wpfloat, **kwargs) -> dict:
        return dict(vapor_deposition_rate=np.full(qi.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            qi               = constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi               = constant_field(grid, 1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            eta              = constant_field(grid, 1.32343e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsi             = constant_field(grid, -0.000618828, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho              = constant_field(grid, 1.19691, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt               = 30.0,
            QMIN             = graupel_ct.qmin,
            vapor_deposition_rate = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )
