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
from icon4py.model.testing import stencil_tests


class TestRainToGraupelDefault(stencil_tests.StencilTest):
    PROGRAM = rain_to_graupel
    OUTPUTS = ("conversion_rate",)

    @stencil_tests.static_reference
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
        return dict(conversion_rate=np.full(t.shape, 0.0))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.12442, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=self.data_alloc.constant_field(1.34006e-17, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=self.data_alloc.constant_field(2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=self.data_alloc.constant_field(1.02627e-19, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi=self.data_alloc.constant_field(1.0e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=self.data_alloc.constant_field(
                -0.000635669, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dt=30.0,
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
