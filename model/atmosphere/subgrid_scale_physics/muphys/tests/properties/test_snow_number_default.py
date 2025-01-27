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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import snow_number
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn, idx

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestSnowNumberDefault(StencilTest):
    PROGRAM = snow_number
    OUTPUTS = ("snow_number",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qs: np.array, QMIN: wpfloat, AMS: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(snow_number=np.full(t.shape, 8.00e+5 ))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t               = constant_field(grid, 276.302, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho             = constant_field(grid, 1.17797, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs              = constant_field(grid, 8.28451e-24, dims.CellDim, dims.KDim, dtype=wpfloat),
            QMIN            = graupel_ct.qmin,
            AMS             = graupel_ct.ams,
            TMELT           = thermodyn.tmelt,
            snow_number     = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
