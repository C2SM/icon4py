# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import rain_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestRainToGraupel1(StencilTest):
    PROGRAM = rain_to_graupel
    OUTPUTS = ("conversion_rate",)

    @static_reference
    def reference(
        grid: base.Grid,
        t: np.ndarray,
        rho: np.ndarray,
        qc: np.ndarray,
        qr: np.ndarray,
        qi: np.ndarray,
        qs: np.ndarray,
        mi: np.ndarray,
        dvsw: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(conversion_rate=np.full(t.shape, 1.0044914238516472e-12))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(230.542, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(0.956089, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.constant_field(8.6157e-5, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=self.data_alloc.constant_field(3.01332e-11, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=self.data_alloc.constant_field(5.57166e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=self.data_alloc.constant_field(3.55432e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi=self.data_alloc.constant_field(1.0e-9, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
