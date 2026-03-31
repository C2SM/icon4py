# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import rain_to_vapor
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestRainToVapor(StencilTest):
    PROGRAM = rain_to_vapor
    OUTPUTS = ("conversion_rate",)

    @static_reference
    def reference(
        grid: base.Grid,
        t: np.ndarray,
        rho: np.ndarray,
        qc: np.ndarray,
        qr: np.ndarray,
        dvsw: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(conversion_rate=np.full(t.shape, 2.8556697055499901e-19))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(258.542, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(0.956089, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=self.data_alloc.constant_field(3.01332e-11, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=self.data_alloc.constant_field(-1.0e-10, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
