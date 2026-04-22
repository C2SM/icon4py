# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import snow_to_rain
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestSnowToRainDefault(stencil_tests.StencilTest):
    PROGRAM = snow_to_rain
    OUTPUTS = ("conversion_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        t: np.ndarray,
        p: np.ndarray,
        rho: np.ndarray,
        dvsw0: np.ndarray,
        qs: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(conversion_rate=np.full(t.shape, 3.7268547760462804e-07))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(275.83, dims.CellDim, dims.KDim, dtype=wpfloat),
            p=self.data_alloc.constant_field(80134.5, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.04892, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0=self.data_alloc.constant_field(
                0.00258631, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qs=self.data_alloc.constant_field(1.47687e-6, dims.CellDim, dims.KDim, dtype=wpfloat),
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
