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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import vel_scale_factor_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn, idx

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat

class TestVelScaleFactorSnow(StencilTest):
    PROGRAM = vel_scale_factor_snow
    OUTPUTS = ("scale_factor",)

    @staticmethod
    def reference(grid, xrho: np.array, **kwargs) -> dict:
        return dict(scale_factor=np.full(xrho.shape, 0.06633230453931642 ))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            xrho            = constant_field(grid, 1.17787, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho             = constant_field(grid, 0.882961, dims.CellDim, dims.KDim, dtype=wpfloat),
            t               = constant_field(grid, 257.101, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs              = constant_field(grid, 5.78761e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            QMIN            = graupel_ct.qmin,
            AMS             = graupel_ct.ams,
            TMELT           = thermodyn.tmelt,
            scale_factor    = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
