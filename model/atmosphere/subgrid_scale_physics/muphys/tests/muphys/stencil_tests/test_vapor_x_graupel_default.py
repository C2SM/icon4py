# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestVaporXGraupelDefault(StencilTest):
    PROGRAM = vapor_x_graupel
    OUTPUTS = ("exchange_rate",)

    @static_reference
    def reference(
        grid: base.Grid,
        t: np.ndarray,
        p: np.ndarray,
        rho: np.ndarray,
        qg: np.ndarray,
        dvsw: np.ndarray,
        dvsi: np.ndarray,
        dvsw0: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(exchange_rate=np.full(t.shape, 0.0))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(278.026, dims.CellDim, dims.KDim, dtype=wpfloat),
            p=self.data_alloc.constant_field(95987.1, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.20041, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=self.data_alloc.constant_field(2.056496e-16, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=self.data_alloc.constant_field(
                -0.00234674, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dvsi=self.data_alloc.constant_field(
                -0.00261576, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dvsw0=self.data_alloc.constant_field(
                -0.00076851, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dt=30.0,
            exchange_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
