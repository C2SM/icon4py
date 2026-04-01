# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import GraupelCt
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import snow_lambda
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestSnowLambda(stencil_tests.StencilTest):
    PROGRAM = snow_lambda
    OUTPUTS = ("riming_snow_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid, rho: np.ndarray, qs: np.ndarray, ns: np.ndarray, **kwargs
    ) -> dict:
        return dict(riming_snow_rate=np.full(rho.shape, 1.0e10))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            rho=self.data_alloc.constant_field(1.12204, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=self.data_alloc.constant_field(
                GraupelCt.qmin, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            ns=self.data_alloc.constant_field(1.76669e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
