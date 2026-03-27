# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import graupel_to_rain
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestGraupelToRain(StencilTest):
    PROGRAM = graupel_to_rain
    OUTPUTS = ("rain_rate",)

    @static_reference
    def reference(
        grid,
        t: np.ndarray,
        p: np.ndarray,
        rho: np.ndarray,
        dvsw0: np.ndarray,
        qg: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(rain_rate=np.full(t.shape, 5.9748142538569357e-13))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            t=self.data_alloc.constant_field(280.156, dims.CellDim, dims.KDim, dtype=wpfloat),
            p=self.data_alloc.constant_field(98889.4, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.22804, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0=self.data_alloc.constant_field(
                -0.00167867, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qg=self.data_alloc.constant_field(1.53968e-15, dims.CellDim, dims.KDim, dtype=wpfloat),
            rain_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
