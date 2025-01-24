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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import fall_speed, fall_speed_scalar
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn, idx

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestFallSpeed(StencilTest):
    PROGRAM = fall_speed_scalar
    OUTPUTS = ("fall_speed")

    @staticmethod
    def reference(grid, density: wpfloat, prefactor: wpfloat, offset: wpfloat, exponent: wpfloat, QMIN: wpfloat, V0S: wpfloat, V1S: wpfloat,  **kwargs) -> dict:
#        return dict(fall_speed=0.67882452435647411)
#        return dict(fall_speed=np.full(density.shape, 0.67882452435647411))
        return dict(fall_speed=np.asarray(0.67882452435647411))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            density         = gtx.as_field([], np.asarray(0.0) ),
#            density         = constant_field(grid, 0.0, 0, 0, dtype=wpfloat),
#            density         = 0.0,
            prefactor       = idx.prefactor_r,
            offset          = idx.offset_r,
            exponent        = idx.exponent_r,
            QMIN            = graupel_ct.qmin,
            V0S             = graupel_ct.v0s,
            V1S             = graupel_ct.v1s,
#            fall_speed      = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat),
#            fall_speed      = 0.0,
            fall_speed      = gtx.as_field([], np.asarray(0.0) ),
        )
