# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestVaporXSnow(StencilTest):
    PROGRAM = vapor_x_snow
    OUTPUTS = ("vapor_deposition_rate",)

    @staticmethod
    def reference(grid, t: np.array, p: np.array, rho: np.array, qs: np.array, ns: np.array, lam: np.array, eta: np.array, ice_dep: np.array, dvsw: np.array, dvsi: np.array, dvsw0: np.array, dt: wpfloat, **kwargs) -> dict:
        return dict(vapor_deposition_rate=np.full(t.shape, -8.6584296264775935e-13))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 278.748, dims.CellDim, dims.KDim, dtype=wpfloat),
            p                = constant_field(grid, 95995.5, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho              = constant_field(grid, 1.19691, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs               = constant_field(grid, 1.25653e-10, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns               = constant_field(grid, 800000.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam              = constant_field(grid, 1.0e10, dims.CellDim, dims.KDim, dtype=wpfloat),
            eta              = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            ice_dep          = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw             = constant_field(grid, -0.00196781, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsi             = constant_field(grid, -0.00229367, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0            = constant_field(grid, -0.000110022, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt               = 30.0,
            vapor_deposition_rate = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

