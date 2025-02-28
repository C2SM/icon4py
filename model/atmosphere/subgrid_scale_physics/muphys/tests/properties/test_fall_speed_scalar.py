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
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import idx
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.helpers import StencilTest

class TestFallSpeed(StencilTest):
    PROGRAM = fall_speed_scalar
    OUTPUTS = ("speed",)

    @staticmethod
    def reference(grid, density: wpfloat, prefactor: wpfloat, offset: wpfloat, exponent: wpfloat, **kwargs) -> dict:
        return dict(speed=np.asarray(0.67882452435647411))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            density         = gtx.as_field([], np.asarray(0.0) ),
            prefactor       = idx.prefactor_r,
            offset          = idx.offset_r,
            exponent        = idx.exponent_r,
            speed           = gtx.as_field([], np.asarray(0.0) ),
        )
