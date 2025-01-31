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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import T_from_internal_energy_scalar

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestTFromInternalEnergy(StencilTest):
    PROGRAM = T_from_internal_energy_scalar
    OUTPUTS = ("temperature",)

    @staticmethod
    def reference(grid, u: wpfloat, qv: wpfloat, qliq: wpfloat, qice: wpfloat, rho: wpfloat, dz: wpfloat, **kwargs) -> dict:
        return dict(temperature=np.asarray(255.75599999999997))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            u                = gtx.as_field([], np.asarray(38265357.270336017) ),
            qv               = gtx.as_field([], np.asarray(0.00122576) ),
            qliq             = gtx.as_field([], np.asarray(1.63837e-20) ),
            qice             = gtx.as_field([], np.asarray(1.09462e-08) ),
            rho              = gtx.as_field([], np.asarray(0.83444) ),
            dz               = gtx.as_field([], np.asarray(249.569) ),
            temperature      = gtx.as_field([], np.asarray(0.0) ),
        )
