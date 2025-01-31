# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import snow_lambda
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestSnowLambda(StencilTest):
    PROGRAM = snow_lambda
    OUTPUTS = ("riming_snow_rate",)

    @staticmethod
    def reference(grid, rho: np.array, qs: np.array, ns: np.array, **kwargs) -> dict:
        return dict(riming_snow_rate=np.full(rho.shape, 1.0e+10 ))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            rho             = constant_field(grid, 1.12204, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs              = constant_field(grid, graupel_ct.qmin, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns              = constant_field(grid, 1.76669e+07, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate= constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
